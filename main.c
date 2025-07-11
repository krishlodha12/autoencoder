/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2015 Intel Corporation
 */

// File: examples/rxtx_callbacks/main.c
// Updated to include daemonization and SQLite logging

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdint.h>
#include <inttypes.h>
#include <rte_common.h>
#include <rte_eal.h>
#include <rte_mbuf.h>
#include <rte_ethdev.h>
#include <rte_malloc.h>
#include <rte_ring.h>
#include <rte_byteorder.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_ip.h>
#include <sqlite3.h>
#include <netinet/ip.h>


#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32
#define MAX_PAYLOAD_SIZE 256

static sqlite3 *db = NULL;

static uint16_t add_timestamps(uint16_t port, uint16_t queue, struct rte_mbuf **pkts,
                                uint16_t nb_pkts, uint16_t max_pkts, void *arg) {
    for (int i = 0; i < nb_pkts; i++) {
        struct rte_mbuf *m = pkts[i];
        uint16_t pkt_len = rte_pktmbuf_pkt_len(m);
        if (pkt_len == 0) continue;

        const uint8_t *data = rte_pktmbuf_mtod(m, const uint8_t *);
        struct rte_ether_hdr *eth_hdr = (struct rte_ether_hdr *)data;

        char src_mac[18], dst_mac[18];
        snprintf(src_mac, sizeof(src_mac), "%02x:%02x:%02x:%02x:%02x:%02x",
                 eth_hdr->src_addr.addr_bytes[0], eth_hdr->src_addr.addr_bytes[1],
                 eth_hdr->src_addr.addr_bytes[2], eth_hdr->src_addr.addr_bytes[3],
                 eth_hdr->src_addr.addr_bytes[4], eth_hdr->src_addr.addr_bytes[5]);

        snprintf(dst_mac, sizeof(dst_mac), "%02x:%02x:%02x:%02x:%02x:%02x",
                 eth_hdr->dst_addr.addr_bytes[0], eth_hdr->dst_addr.addr_bytes[1],
                 eth_hdr->dst_addr.addr_bytes[2], eth_hdr->dst_addr.addr_bytes[3],
                 eth_hdr->dst_addr.addr_bytes[4], eth_hdr->dst_addr.addr_bytes[5]);

        uint64_t timestamp = rte_get_tsc_cycles();

        // Classify protocol
        uint16_t ether_type = rte_be_to_cpu_16(eth_hdr->ether_type);
        char proto_name[32];

        if (ether_type == RTE_ETHER_TYPE_IPV4) {
            if (pkt_len >= sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr)) {
                struct rte_ipv4_hdr *ipv4_hdr = (struct rte_ipv4_hdr *)(data + sizeof(struct rte_ether_hdr));
                uint8_t proto = ipv4_hdr->next_proto_id;

                if (proto == IPPROTO_ICMP) {
                    snprintf(proto_name, sizeof(proto_name), "_icmp");
                } else if (proto == IPPROTO_UDP) {
                    snprintf(proto_name, sizeof(proto_name), "_udp");
                } else if (proto == IPPROTO_TCP) {
                    snprintf(proto_name, sizeof(proto_name), "_tcp");
                } else {
                    snprintf(proto_name, sizeof(proto_name), "_ipv4");
                }
            } else {
                snprintf(proto_name, sizeof(proto_name), "_ipv4_truncated");
            }
        } else if (ether_type == RTE_ETHER_TYPE_IPV6) {
            snprintf(proto_name, sizeof(proto_name), "_ipv6");
        } else if (ether_type == RTE_ETHER_TYPE_ARP) {
            snprintf(proto_name, sizeof(proto_name), "_arp");
        } else {
            snprintf(proto_name, sizeof(proto_name), "_not_ipv4_ipv6");
        }

        if (db) {
            char *err_msg = NULL;

            // Update protocol count
            char sql_proto[256];
            snprintf(sql_proto, sizeof(sql_proto),
                     "INSERT INTO protocols (name, count) VALUES ('%s', 1) "
                     "ON CONFLICT(name) DO UPDATE SET count = count + 1;", proto_name);
            sqlite3_exec(db, sql_proto, 0, 0, &err_msg);
            if (err_msg) {
                fprintf(stderr, "SQLite error (protocols): %s\n", err_msg);
                sqlite3_free(err_msg);
            }

        }
    }

    return nb_pkts;
}
int main(int argc, char *argv[]) {
    printf("Calling rte_eal_init...\n");
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) rte_exit(EXIT_FAILURE, "EAL init failed\n");

    printf("EAL initialized successfully. Returned %d\n", ret);
    printf("Remaining argc: %d\n", argc - ret);
    for (int i = ret; i < argc; i++) {
        printf("Remaining arg[%d]: %s\n", i, argv[i]);
    }

    pid_t pid = fork();
    if (pid < 0) exit(EXIT_FAILURE);
    if (pid > 0) exit(EXIT_SUCCESS);
    setsid();

    int fd = open("/dev/null", O_RDWR);
    if (fd > 0) {
        dup2(fd, STDIN_FILENO);
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        if (fd > 2) close(fd);
    }

    uint16_t nb_ports = rte_eth_dev_count_avail();
    if (nb_ports < 1) rte_exit(EXIT_FAILURE, "No available DPDK ports found\n");
    printf("DPDK reports %u port(s) available\n", nb_ports);

    struct rte_mempool *mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL",
        NUM_MBUFS * nb_ports, MBUF_CACHE_SIZE, 0,
        RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());

    if (mbuf_pool == NULL)
        rte_exit(EXIT_FAILURE, "mbuf pool creation failed\n");

    if (sqlite3_open("/var/log/dpdk_rxlog.db", &db) != SQLITE_OK) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        return 1;
    }

    const char *create_packets_sql =
        "CREATE TABLE IF NOT EXISTS packets ("
        "src_mac TEXT,"
        "dst_mac TEXT,"
        "count INTEGER,"
        "PRIMARY KEY (src_mac, dst_mac));";

    const char *create_payloads_sql =
        "CREATE TABLE IF NOT EXISTS payloads ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "timestamp INTEGER,"
        "payload_hex TEXT);";

    const char *create_framesize_sql =
        "CREATE TABLE IF NOT EXISTS framesize ("
        "name TEXT PRIMARY KEY,"
        "count INTEGER);";

    char *err_msg = NULL;
    if (sqlite3_exec(db, create_packets_sql, 0, 0, &err_msg) != SQLITE_OK) {
        fprintf(stderr, "SQL error (packets table): %s\n", err_msg);
        sqlite3_free(err_msg);
    }

    const char *create_protocols_sql =
        "CREATE TABLE IF NOT EXISTS protocols ("
        "name TEXT PRIMARY KEY,"
        "count INTEGER);";

    if (sqlite3_exec(db, create_protocols_sql, 0, 0, &err_msg) != SQLITE_OK) {
        fprintf(stderr, "SQL error (protocols table): %s\n", err_msg);
        sqlite3_free(err_msg);
    }

    if (sqlite3_exec(db, create_payloads_sql, 0, 0, &err_msg) != SQLITE_OK) {
        fprintf(stderr, "SQL error (payloads table): %s\n", err_msg);
        sqlite3_free(err_msg);
    }
    if (sqlite3_exec(db, create_framesize_sql, 0, 0, &err_msg) != SQLITE_OK) {
        fprintf(stderr, "SQL error (framesize table): %s\n", err_msg);
        sqlite3_free(err_msg);
    }

    struct rte_eth_conf port_conf = {0};
    uint16_t portid = 0, active_port = 0;
    RTE_ETH_FOREACH_DEV(portid) {
        printf("Initializing port %u\n", portid);
        rte_eth_dev_configure(portid, 1, 1, &port_conf);
        rte_eth_rx_queue_setup(portid, 0, RX_RING_SIZE, rte_eth_dev_socket_id(portid), NULL, mbuf_pool);
        rte_eth_tx_queue_setup(portid, 0, TX_RING_SIZE, rte_eth_dev_socket_id(portid), NULL);
        rte_eth_dev_start(portid);
        rte_eth_promiscuous_enable(portid);

        struct rte_eth_link link;
        rte_eth_link_get_nowait(portid, &link);
        printf("Port %u link status: %s, speed %u Mbps\n",
               portid, link.link_status ? "UP" : "DOWN", link.link_speed);

        printf("Setting Rx callback on port %d\n", portid);
        rte_eth_add_rx_callback(portid, 0, add_timestamps, NULL);

        active_port = portid;
    }

    while (1) {
        struct rte_mbuf *bufs[BURST_SIZE];
        uint16_t nb_rx = rte_eth_rx_burst(active_port, 0, bufs, BURST_SIZE);
        if (nb_rx > 0) {
            for (int i = 0; i < nb_rx; i++) {
                rte_pktmbuf_free(bufs[i]);
            }
        }
        rte_delay_ms(100);
    }

    if (db) sqlite3_close(db);
    rte_eal_cleanup();
    return 0;
}
