/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2015 Intel Corporation
 */

// File: examples/rxtx_callbacks/main.c
// Updated to include daemonization and SQLite logging

#include <stdalign.h>
#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>
#include <getopt.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_mbuf.h>
#include <rte_mbuf_dyn.h>
#include <rte_ether.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <sqlite3.h>

#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32

static int hwts_dynfield_offset = -1;

static inline rte_mbuf_timestamp_t *
hwts_field(struct rte_mbuf *mbuf)
{
    return RTE_MBUF_DYNFIELD(mbuf, hwts_dynfield_offset, rte_mbuf_timestamp_t *);
}

typedef uint64_t tsc_t;
static int tsc_dynfield_offset = -1;

static inline tsc_t *
tsc_field(struct rte_mbuf *mbuf)
{
    return RTE_MBUF_DYNFIELD(mbuf, tsc_dynfield_offset, tsc_t *);
}

static const char usage[] =
    "%s EAL_ARGS -- [-t]\n";

static struct {
    uint64_t total_cycles;
    uint64_t total_queue_cycles;
    uint64_t total_pkts;
} latency_numbers;

static int hw_timestamping;
#define TICKS_PER_CYCLE_SHIFT 16
static uint64_t ticks_per_cycle_mult;

static sqlite3 *db = NULL;

/* Callback added to the RX port and applied to packets. 8< */
static uint16_t
add_timestamps(uint16_t port __rte_unused, uint16_t qidx __rte_unused,
        struct rte_mbuf **pkts, uint16_t nb_pkts,
        uint16_t max_pkts __rte_unused, void * __rte_unused)
{
    unsigned i;
    uint64_t now = rte_rdtsc();

    for (i = 0; i < nb_pkts; i++) {
        *tsc_field(pkts[i]) = now;

        struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(pkts[i], struct rte_ether_hdr *);
        struct rte_ether_addr *src = &eth_hdr->src_addr;
        struct rte_ether_addr *dst = &eth_hdr->dst_addr;

        char src_mac[18], dst_mac[18];
        snprintf(src_mac, sizeof(src_mac), "%02x:%02x:%02x:%02x:%02x:%02x",
                 src->addr_bytes[0], src->addr_bytes[1], src->addr_bytes[2],
                 src->addr_bytes[3], src->addr_bytes[4], src->addr_bytes[5]);
        snprintf(dst_mac, sizeof(dst_mac), "%02x:%02x:%02x:%02x:%02x:%02x",
                 dst->addr_bytes[0], dst->addr_bytes[1], dst->addr_bytes[2],
                 dst->addr_bytes[3], dst->addr_bytes[4], dst->addr_bytes[5]);

        if (db) {
            char *err_msg = NULL;
            char sql[256];
            snprintf(sql, sizeof(sql),
                     "INSERT INTO packets (timestamp, src_mac, dst_mac) VALUES (%" PRIu64 ", '%s', '%s');",
                     now, src_mac, dst_mac);
            sqlite3_exec(db, sql, 0, 0, &err_msg);
            if (err_msg) {
                fprintf(stderr, "SQLite insert error: %s\n", err_msg);
                sqlite3_free(err_msg);
            }
        }
    }

    return nb_pkts;
}
/* >8 End of callback addition and application. */

int main(int argc, char *argv[])
{
    //pid_t pid = fork();
    //if (pid < 0) {
      //  perror("fork failed");
        //exit(EXIT_FAILURE);
    //}
    //if (pid > 0) {
      //  exit(EXIT_SUCCESS);
    //}

    //if (setsid() < 0) {
      //  perror("setsid failed");
        //exit(EXIT_FAILURE);
    //}

    int fd = open("/dev/null", O_RDWR, 0);
    if (fd != -1) {
        dup2(fd, STDIN_FILENO);
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        if (fd > 2) close(fd);
    }

    struct rte_mempool *mbuf_pool;
    uint16_t nb_ports;
    uint16_t portid;
    struct option lgopts[] = {
        { NULL, 0, 0, 0 }
    };
    int opt, option_index;

    static const struct rte_mbuf_dynfield tsc_dynfield_desc = {
        .name = "example_bbdev_dynfield_tsc",
        .size = sizeof(tsc_t),
        .align = alignof(tsc_t),
    };

    int ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");

    argc -= ret;
    argv += ret;

    while ((opt = getopt_long(argc, argv, "t", lgopts, &option_index)) != -1) {
        switch (opt) {
        case 't':
            hw_timestamping = 1;
            break;
        default:
            printf(usage, argv[0]);
            return -1;
        }
    }

    optind = 1;

    nb_ports = rte_eth_dev_count_avail();
    if (nb_ports < 2 || (nb_ports & 1))
        rte_exit(EXIT_FAILURE, "Error: number of ports must be even\n");

    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS * nb_ports,
        MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (mbuf_pool == NULL)
        rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

    tsc_dynfield_offset = rte_mbuf_dynfield_register(&tsc_dynfield_desc);
    if (tsc_dynfield_offset < 0)
        rte_exit(EXIT_FAILURE, "Cannot register mbuf field\n");

    // Open SQLite DB in /var/log
    int db_rc = sqlite3_open("/home/lodha/dpdk_rxlog.db", &db);
    if (db_rc != SQLITE_OK) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        return 1;
    }

    const char *create_table_sql =
        "CREATE TABLE IF NOT EXISTS packets ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "timestamp INTEGER,"
        "src_mac TEXT,"
        "dst_mac TEXT);";

    char *err_msg = NULL;
    db_rc = sqlite3_exec(db, create_table_sql, 0, 0, &err_msg);
    if (db_rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
        sqlite3_close(db);
        return 1;
    }

    // Proceed with port init and callback setup here...

    rte_eal_cleanup();
    if (db) sqlite3_close(db);
    return 0;
}
