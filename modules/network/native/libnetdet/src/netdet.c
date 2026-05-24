/**
 * libnetdet — DPDK transmit/receive with SHA-256 digest.
 *
 * Thin wrapper: Python builds frames, this library transmits them
 * via kernel bypass and computes integrity digests.
 */
#include "netdet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_cycles.h>
#include <rte_timer.h>

#include <openssl/evp.h>

#define POOL_SIZE 8191
#define POOL_CACHE 250
#define RX_RING_SIZE 512
#define TX_RING_SIZE 512

struct netdet_ctx {
    uint16_t port_id;
    struct rte_mempool* pool;
};

netdet_ctx* netdet_init(int argc, char** argv, uint16_t port_id) {
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        fprintf(stderr, "netdet: rte_eal_init failed: %s\n",
                rte_strerror(rte_errno));
        return NULL;
    }

    /* Create mempool */
    struct rte_mempool* pool = rte_pktmbuf_pool_create(
        "NETDET_POOL", POOL_SIZE, POOL_CACHE, 0,
        RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (!pool) {
        fprintf(stderr, "netdet: mempool creation failed: %s\n",
                rte_strerror(rte_errno));
        rte_eal_cleanup();
        return NULL;
    }

    /* Configure port: disable ALL offloads */
    struct rte_eth_conf port_conf = {
        .rxmode = { .mq_mode = RTE_ETH_MQ_RX_NONE, .offloads = 0 },
        .txmode = { .mq_mode = RTE_ETH_MQ_TX_NONE, .offloads = 0 },
    };

    ret = rte_eth_dev_configure(port_id, 1, 1, &port_conf);
    if (ret < 0) {
        fprintf(stderr, "netdet: port configure failed: %s\n",
                rte_strerror(-ret));
        rte_eal_cleanup();
        return NULL;
    }

    /* Setup TX queue */
    ret = rte_eth_tx_queue_setup(port_id, 0, TX_RING_SIZE,
                                  rte_eth_dev_socket_id(port_id), NULL);
    if (ret < 0) {
        fprintf(stderr, "netdet: TX queue setup failed: %s\n",
                rte_strerror(-ret));
        rte_eal_cleanup();
        return NULL;
    }

    /* Setup RX queue */
    ret = rte_eth_rx_queue_setup(port_id, 0, RX_RING_SIZE,
                                  rte_eth_dev_socket_id(port_id), NULL, pool);
    if (ret < 0) {
        fprintf(stderr, "netdet: RX queue setup failed: %s\n",
                rte_strerror(-ret));
        rte_eal_cleanup();
        return NULL;
    }

    /* Start port */
    ret = rte_eth_dev_start(port_id);
    if (ret < 0) {
        fprintf(stderr, "netdet: port start failed: %s\n",
                rte_strerror(-ret));
        rte_eal_cleanup();
        return NULL;
    }

    /* Enable promiscuous mode for loopback capture */
    rte_eth_promiscuous_enable(port_id);

    /* Log offload status */
    struct rte_eth_dev_info dev_info;
    rte_eth_dev_info_get(port_id, &dev_info);
    fprintf(stderr, "netdet: port %u started. TX offload capa: 0x%lx\n",
            port_id, (unsigned long)dev_info.tx_offload_capa);

    netdet_ctx* ctx = calloc(1, sizeof(netdet_ctx));
    if (!ctx) {
        rte_eth_dev_stop(port_id);
        rte_eth_dev_close(port_id);
        rte_eal_cleanup();
        return NULL;
    }
    ctx->port_id = port_id;
    ctx->pool = pool;
    return ctx;
}

netdet_tx_result netdet_send(netdet_ctx* ctx,
                             const uint8_t** frames,
                             const uint16_t* lengths,
                             int count) {
    netdet_tx_result result;
    memset(&result, 0, sizeof(result));

    if (!ctx || count <= 0) return result;

    /* Allocate mbufs on heap for large counts */
    struct rte_mbuf** mbufs = calloc(count, sizeof(struct rte_mbuf*));
    if (!mbufs) return result;

    for (int i = 0; i < count; i++) {
        mbufs[i] = rte_pktmbuf_alloc(ctx->pool);
        if (!mbufs[i]) {
            fprintf(stderr, "netdet: mbuf alloc failed at frame %d\n", i);
            for (int j = 0; j < i; j++) rte_pktmbuf_free(mbufs[j]);
            free(mbufs);
            result.submitted = i;
            result.confirmed = 0;
            return result;
        }
        char* data = rte_pktmbuf_append(mbufs[i], lengths[i]);
        memcpy(data, frames[i], lengths[i]);
        /* Do NOT set any mbuf offload flags (ol_flags stays 0) */
    }

    result.submitted = count;

    EVP_MD_CTX* sha_ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(sha_ctx, EVP_sha256(), NULL);

    /* Transmit burst */
    uint16_t sent = rte_eth_tx_burst(ctx->port_id, 0,
                                      mbufs, (uint16_t)count);

    /*
     * Hash the bytes of mbufs that were accepted by tx_burst.
     * For mlx5, the PMD copies data into the WQE during tx_burst,
     * so the mbuf content at this point is the source of truth.
     */
    for (uint16_t i = 0; i < sent; i++) {
        EVP_DigestUpdate(sha_ctx,
                         rte_pktmbuf_mtod(mbufs[i], uint8_t*),
                         rte_pktmbuf_pkt_len(mbufs[i]));
    }

    /* Free unsent mbufs (sent ones are freed by the driver after TX) */
    for (uint16_t i = sent; i < (uint16_t)count; i++) {
        rte_pktmbuf_free(mbufs[i]);
    }

    result.confirmed = sent;
    unsigned int digest_len = 0;
    EVP_DigestFinal_ex(sha_ctx, result.digest, &digest_len);
    EVP_MD_CTX_free(sha_ctx);
    free(mbufs);
    return result;
}

netdet_rx_result netdet_recv(netdet_ctx* ctx, int timeout_ms) {
    netdet_rx_result result;
    memset(&result, 0, sizeof(result));

    if (!ctx || timeout_ms <= 0) return result;

    EVP_MD_CTX* sha_ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(sha_ctx, EVP_sha256(), NULL);

    struct rte_mbuf* mbufs[256];
    int total = 0;
    int capacity = 256;
    result.frames = malloc(capacity * sizeof(uint8_t*));
    result.lengths = malloc(capacity * sizeof(uint16_t));

    uint64_t hz = rte_get_timer_hz();
    uint64_t deadline = rte_get_timer_cycles()
                      + (uint64_t)timeout_ms * hz / 1000;

    while (rte_get_timer_cycles() < deadline) {
        uint16_t nb = rte_eth_rx_burst(ctx->port_id, 0, mbufs, 256);
        for (uint16_t i = 0; i < nb; i++) {
            uint16_t len = rte_pktmbuf_pkt_len(mbufs[i]);
            uint8_t* copy = malloc(len);
            memcpy(copy, rte_pktmbuf_mtod(mbufs[i], uint8_t*), len);

            if (total >= capacity) {
                capacity *= 2;
                result.frames = realloc(result.frames,
                                        capacity * sizeof(uint8_t*));
                result.lengths = realloc(result.lengths,
                                         capacity * sizeof(uint16_t));
            }
            result.frames[total] = copy;
            result.lengths[total] = len;
            total++;

            EVP_DigestUpdate(sha_ctx, copy, len);
            rte_pktmbuf_free(mbufs[i]);
        }
        if (nb == 0) {
            rte_delay_us_block(100);
        }
    }

    result.count = total;
    unsigned int digest_len = 0;
    EVP_DigestFinal_ex(sha_ctx, result.digest, &digest_len);
    EVP_MD_CTX_free(sha_ctx);
    return result;
}

void netdet_rx_free(netdet_rx_result* result) {
    if (!result) return;
    for (int i = 0; i < result->count; i++) {
        free(result->frames[i]);
    }
    free(result->frames);
    free(result->lengths);
    result->frames = NULL;
    result->lengths = NULL;
    result->count = 0;
}

void netdet_close(netdet_ctx* ctx) {
    if (!ctx) return;
    rte_eth_dev_stop(ctx->port_id);
    rte_eth_dev_close(ctx->port_id);
    /* rte_eal_cleanup() can only be called once; skip if already called */
    rte_eal_cleanup();
    free(ctx);
}
