import argparse
import csv
import logging
from pathlib import Path

from scapy.all import *

# These layers need to be loaded manually it seems
load_layer("http")


# load_layer("ftp")
# load_layer("smtp")
# check if layer 7 is one of these protocol type, I could not find an easier way to get layer name
def check_layer_7_proto(packet):
    if packet.haslayer(DNS):
        return "DNS"
    elif packet.haslayer(HTTP):
        return "HTTP"
    # elif packet.haslayer(FTP):
    #    return "FTP"
    # elif packet.haslayer(SMTP):
    #    return "SMTP"
    else:
        return "nil"


def parse_pcap(pcap_file: Path, output_path: Path):

    if not pcap_file:
        logging.error("no file path found")
        return None

    logging.info("reading file")
    pkts = rdpcap(str(pcap_file.absolute()))

    logging.info("parsing packets")
    data = []
    for pkt in pkts:
        time = pkt.time

        if not pkt.haslayer(IP):
            continue

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst

        if pkt.haslayer(TCP):
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            layer4_protocol = "TCP"
        elif pkt.haslayer(UDP):
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
            layer4_protocol = "UDP"
        else:
            src_port = ""
            dst_port = ""
            layer4_protocol = ""

        l7_proto = check_layer_7_proto(pkt)

        print(time, src_ip, dst_ip, src_port, dst_port, layer4_protocol, l7_proto)

        # Get the layer 2 protocol information
        if pkt.haslayer(Ether):
            layer2_protocol = pkt[Ether].type
        else:
            layer2_protocol = ""

        # Get the layer 7 protocol information
        # if pkt.haslayer(Raw):
        #    layer7_protocol = pkt[Raw].load
        # else:
        #    layer7_protocol = ''

        # Append the data to the list
        data.append(
            [
                time,
                src_ip,
                src_port,
                dst_ip,
                dst_port,
                layer4_protocol,
                layer2_protocol,
                l7_proto,
            ]
        )

    # Create the column names
    column_names = [
        "timestamp",
        "source_ip",
        "source_port",
        "destination_ip",
        "destination_port",
        "l4_proto",
        "l2_proto",
        "l7_proto",
    ]
    logging.info("dumping data to file")
    # Save the data to a CSV file
    output_file = str(output_path.absolute())
    print("Dumping csv in", output_file)
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
        writer.writerows(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="--input <path to input pcap file>", type=str)
    parser.add_argument("--output", help="--output <path to output csv file>", type=str)
    args = parser.parse_args()

    parse_pcap(Path(args.input), Path(args.output))


main()
