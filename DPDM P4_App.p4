/*************************************************************************
*********************** DPDM Module  ***********************************
*************************************************************************/

// P4 program for Data-plane-based Dynamic Monitoring (DPDM)

#include <core.p4>
#include <v1model.p4>

header ethernet_t {
    bit dstAddr;
    bit srcAddr;
    bit etherType;
}

header ipv4_t {
    bit version;
    bit ihl;
    bit diffserv;
    bit totalLen;
    bit identification;
    bit flags;
    bit fragOffset;
    bit ttl;
    bit protocol;
    bit hdrChecksum;
    bit srcAddr;
    bit dstAddr;
}

header tcp_t {
    bit srcPort;
    bit dstPort;
    bit seqNo;
    bit ackNo;
    bit dataOffset;
    bit reserved;
    bit flags;
    bit window;
    bit checksum;
    bit urgentPtr;
}

struct headers {
    ethernet_t ethernet;
    ipv4_t ipv4;
    tcp_t tcp;
}

struct metadata {
    // Add metadata for dynamic windowing if needed
    bit flow_id;
    bit packet_length;
    bit time_stamp;
}

parser MyParser(packet_in b, out headers hdr, out metadata meta) {
    extract(hdr.ethernet);
    if (hdr.ethernet.etherType == 0x0800) {
        extract(hdr.ipv4);
        if (hdr.ipv4.protocol == 6) { // TCP
            extract(hdr.tcp);
        }
    }
    meta.packet_length = b.length;
    meta.time_stamp = timeSinceBoot(); // Example time stamp
    meta.flow_id = hdr.ipv4.srcAddr + hdr.ipv4.dstAddr + hdr.tcp.srcPort + hdr.tcp.dstPort;
}


control Ingress(inout headers hdr, inout metadata meta, inout standard_metadata_t std_meta) {

    // Phase 1: State Tables (Example - simplified for demonstration)
    table flow_stats {
        key {
            meta.flow_id: exact;
        }
        actions {
            count;
            no_op;
        }
        size: 1024; // Example size
    }

    // Phase 3: P4 Counters and Registers
    Counter packet_counter;
    Counter byte_counter;
    Register<bit> flow_packet_count[1024];

    // Phase 4: P4 Program Implementation (Match-Action)
    apply flow_stats.apply();
    flow_packet_count[meta.flow_id] = flow_packet_count[meta.flow_id] + 1;
    packet_counter.count();
    byte_counter.add(meta.packet_length);

    // Dynamic Windowing Logic (Simplified Example)
    if(flow_packet_count[meta.flow_id] > 1000){ //Example threshold for dynamic windowing
        //Perform some action, like marking the packet or increasing monitoring frequency
        std_meta.egress_spec = 1; //Example action: send to a specific port for further inspection
    }

    
control Egress(inout headers hdr, inout metadata meta, inout standard_metadata_t std_meta) {
    // Egress processing (if needed)
}

control VerifyChecksum(inout headers hdr, inout metadata meta) {
    // Checksum verification (if needed)
}

control ComputeChecksum(inout headers hdr, inout metadata meta) {
    // Checksum computation (if needed)
}

V1Switch(
    parser = MyParser,
    ingress = Ingress,
    egress = Egress,
    verifyChecksum = VerifyChecksum,
    computeChecksum = ComputeChecksum
);
