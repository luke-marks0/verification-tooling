/**
 * libnetdet — thin DPDK wrapper for deterministic frame transmission.
 *
 * This library does NOT build frames. Python does that. It only:
 *   1. Initializes DPDK EAL and opens a port
 *   2. Copies pre-built frame bytes into mbufs and transmits them
 *   3. Computes SHA-256 over transmitted mbuf data for TX completion verification
 *   4. Receives frames from an RX port (for loopback verification)
 *   5. Cleans up
 */
#ifndef NETDET_H
#define NETDET_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque context. */
typedef struct netdet_ctx netdet_ctx;

/** TX result: how many frames confirmed, SHA-256 digest. */
typedef struct {
    int confirmed;           /**< Frames confirmed transmitted. */
    int submitted;           /**< Frames submitted. */
    uint8_t digest[32];      /**< SHA-256 over confirmed frame bytes. */
} netdet_tx_result;

/** RX result: received frames + digest. */
typedef struct {
    int count;               /**< Frames received. */
    uint8_t** frames;        /**< Array of frame pointers (caller frees via netdet_rx_free). */
    uint16_t* lengths;       /**< Array of frame lengths. */
    uint8_t digest[32];      /**< SHA-256 over received frame bytes. */
} netdet_rx_result;

/**
 * Initialize DPDK EAL, set up port, allocate mempool.
 *
 * @param argc  Number of EAL arguments.
 * @param argv  EAL argument array.
 * @param port_id  DPDK port to use.
 * @return Opaque context, or NULL on failure.
 */
netdet_ctx* netdet_init(int argc, char** argv, uint16_t port_id);

/**
 * Transmit pre-built frames, wait for completion, return digest.
 *
 * The digest is computed over the mbuf data immediately after
 * rte_eth_tx_burst(). For mlx5, the PMD copies data into the WQE
 * during tx_burst, so the mbuf content is the source of truth.
 *
 * @param ctx     Context from netdet_init().
 * @param frames  Array of pointers to raw L2 frame bytes.
 * @param lengths Array of frame lengths.
 * @param count   Number of frames.
 * @return TX result with confirmed count and SHA-256 digest.
 */
netdet_tx_result netdet_send(netdet_ctx* ctx,
                             const uint8_t** frames,
                             const uint16_t* lengths,
                             int count);

/**
 * Receive frames (blocking up to timeout_ms).
 *
 * Used for loopback verification (Level 2). Returns all frames
 * received during the timeout window.
 *
 * @param ctx         Context from netdet_init().
 * @param timeout_ms  Maximum time to wait for frames.
 * @return RX result with received frames and SHA-256 digest.
 */
netdet_rx_result netdet_recv(netdet_ctx* ctx, int timeout_ms);

/**
 * Free RX result buffers allocated by netdet_recv().
 */
void netdet_rx_free(netdet_rx_result* result);

/**
 * Tear down DPDK port, free mempool, clean up EAL.
 */
void netdet_close(netdet_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* NETDET_H */
