/**
 * Basic test for libnetdet.
 *
 * Without DPDK hardware: verifies that netdet_init() returns NULL
 * (stubs are not implemented).
 *
 * With DPDK hardware: pass EAL args and a port ID to test init/close.
 *   ./test_netdet --no-huge -l 0 -- 0
 *   (last arg after -- is the port ID)
 */
#include "netdet.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_stub_returns_null(void) {
    /* Stubs always return NULL. */
    char* argv[] = {"test_netdet"};
    netdet_ctx* ctx = netdet_init(1, argv, 0);
    assert(ctx == NULL);
    printf("PASS: stub init returns NULL\n");
}

static void test_rx_free_null_safe(void) {
    /* netdet_rx_free should handle a zeroed result. */
    netdet_rx_result result;
    memset(&result, 0, sizeof(result));
    netdet_rx_free(&result);
    printf("PASS: rx_free handles zeroed result\n");
}

static void test_with_dpdk(int argc, char** argv, uint16_t port_id) {
    netdet_ctx* ctx = netdet_init(argc, argv, port_id);
    assert(ctx != NULL);
    netdet_close(ctx);
    printf("PASS: init/close with DPDK\n");
}

int main(int argc, char** argv) {
    /* Check if DPDK args were provided (separated by --). */
    int separator = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) {
            separator = i;
            break;
        }
    }

    if (separator > 0 && separator + 1 < argc) {
        /* DPDK mode: args before -- are EAL args, arg after -- is port ID. */
        uint16_t port_id = (uint16_t)atoi(argv[separator + 1]);
        test_with_dpdk(separator, argv, port_id);
    } else {
        /* Stub mode: no DPDK, just verify stubs behave. */
        test_stub_returns_null();
        test_rx_free_null_safe();
    }

    return 0;
}
