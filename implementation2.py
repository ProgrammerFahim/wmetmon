import os

os.system('C:\\"Program Files"\Wireshark\\tshark -r test_chopchop_wireshark.pcapng -T fields '
            '-e frame.time_delta -e frame.len -e radiotap.datarate '
            '-e radiotap.channel.flags.cck -e radiotap.dbm_antsignal -e wlan.fc.frag '
            '-e wlan.fc.retry -e wlan.fc.protected -e wlan.duration '
            '-e wlan.frag -e wlan.seq -e wlan.fc.ds -e wlan.fc.type -e wlan.fc.subtype '
            '> chopchop_wireshark.txt')
