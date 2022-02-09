# -_-coding:UTF-8-_-

# ryu-manager app/ofctl_rest.py 流量矩阵.py --observe-links

#sudo mn --topo=tree,2,2 --controller=remote  --mac --link tc,bw=100


import numpy as np

from ryu.base import app_manager

from ryu.ofproto import ofproto_v1_3

from ryu.controller.handler import set_ev_cls

from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER,DEAD_DISPATCHER

from ryu.controller import ofp_event 

from ryu.lib.packet import packet 

from ryu.lib.packet import ethernet

from operator import attrgetter 

from ryu.lib import hub 

from ryu.topology.api import get_all_host, get_all_link, get_all_switch

from ryu.lib.packet import ipv4

from ryu.lib.packet import arp 

from ryu.lib.packet import icmp 

from ryu.lib.packet import ether_types 

from ryu.lib.packet import lldp

from ryu.lib.packet import packet

from ryu.base.app_manager import lookup_service_brick

from ryu.lib import mac 

from ryu.topology.api import get_switch, get_link 

from ryu.app.wsgi import ControllerBase 

from ryu.topology import event, switches 

import networkx as nx

import time

import json

import pandas as pd

import requests

ETHERNET = ethernet.ethernet.__name__

ETHERNET_MULTICAST = "ff:ff:ff:ff:ff:ff"

ARP = arp.arp.__name__

LLDP = lldp.lldp.__name__

LLDPPacket = switches.LLDPPacket

delay = {}                      #交换机之间的链路时延

hard_timeout = -1


class NetworkAwareness(app_manager.RyuApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):

        super(NetworkAwareness, self).__init__(*args, **kwargs)

        self.topology_api_app = self

        self.EPISODE = 3000


        # 网络拓扑

        self.graph = nx.DiGraph()               #有向图  () 交换机之间获取时延添加 "lldpdelay"属性，剩下个节点对之间都存在"weight"属性

        self.datapaths = {}                             #交换机 id 和交换机信息

        self.host_num = 0                             # 主机num

        # 流量矩阵

        self.switch_host_port = {}            #交换机连接主机的端口
        self.flows=[]
        self.weight_length=0
        self.weight_sum=[]


        # 剩余流量

        self.port_stats = {}                           # 将key(交换机，端口) 和 value(接收数据，发送数据，接收错误，时间1，时间2) 添加到 self.port_stats

        self.port_speed = {}                        # 交换机端口流速

        self.dpid_port = {}                           #节点与节点之间端口连接关系

        self.traffic = None

        self.re_num = None
 
        self.avg_rest=[]	
        # 包入操作，储存MAC
        self.mac_table={}
        self.arp_table={}                         #主机IP和MAC地址

        self.datapath_switch = {}           #交换机ID和交换机信息

        # 进程协助

        self.sswitches = lookup_service_brick('switches')

        self.topo_thread = hub.spawn(self._get_topology)
        #self.topo_thread = hub.spawn(self._monitor)			# 最短路径（实验一） 把这行代码注释掉


    @set_ev_cls(ofp_event.EventOFPStateChange,[MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):

        """

            Record datapath's info

            记录datapath的信息

            查看交换机变化

        """

        datapath = ev.datapath

        if ev.state == MAIN_DISPATCHER:

            if not datapath.id in self.datapaths:

                self.logger.debug('register datapath: %016x', datapath.id)

                self.datapaths[datapath.id] = datapath

        elif ev.state == DEAD_DISPATCHER:

            if datapath.id in self.datapaths:

                self.logger.debug('unregister datapath: %016x', datapath.id)

                del self.datapaths[datapath.id]

    def add_flow(self, datapath, priority, match, actions,hard_timeout=0):

        "添加流表"

        dp = datapath

        ofp = dp.ofproto

        parser = dp.ofproto_parser

        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]

        mod = parser.OFPFlowMod(datapath=dp, priority=priority, match=match, instructions=inst,hard_timeout=hard_timeout)

        dp.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):

        "添加默认流表"

        msg = ev.msg

        dp = msg.datapath

        ofp = dp.ofproto

        parser = dp.ofproto_parser

        match = parser.OFPMatch()

        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]

        self.add_flow(dp, 0, match, actions,0)

    #获取网络拓扑

    def _get_topology(self):

        "获取网络节点，构建图"


        hub.sleep(1)

        switch_list = get_switch(self.topology_api_app, None)           # 获取交换机列表

        switches = [switch.dp.id for switch in switch_list]                     # 获取交换机datapath.id

        self.graph.add_nodes_from(switches)                                         #添加节点入图

        link_list = get_link(self.topology_api_app, None)

        for link in link_list:

            key = (link.src.dpid,link.src.port_no)

            self._save_stats(self.dpid_port,key,link.dst.dpid,50)

            self.graph.add_edge(link.src.dpid, link.dst.dpid, weight=1, port=link.src.port_no)

            self.graph.add_edge(link.dst.dpid, link.src.dpid, weight=1, port=link.dst.port_no)          # 连接节点（双向）

        switch_all_port = {}
        print(self.graph.edges(data=True))

        for switch in switch_list:

            # 剔除交换机端口,剩下的就是主机连接端口

            dpid = switch.dp.id

            flag = False

            for port in switch.ports:

                "添加全部交换机端口"

                if flag:

                    switch_all_port[dpid].add(port.port_no)

                    continue    

                if dpid not in switch_all_port:

                    switch_all_port[dpid] = {port.port_no}

                    flag = True

            for link in link_list:

                Src = link.src

                Dst = link.dst

                if Src.dpid in switch_all_port:

                        switch_all_port[Src.dpid].discard(Src.port_no)              # 通过边的连接节点来剔除交换机与交换机之间的端口，那么剩下的端口就是连接终端的

                if Dst.dpid in switch_all_port:

                        switch_all_port[Dst.dpid].discard(Dst.port_no)


    #流量矩阵


    def _monitor(self):
        """
            Main entry method of monitoring traffic.
            监控流量的主要入口方式。
        """
        hub.sleep(1)
        self.link_bw_dict_o=[[1,100,100,1,1,1,1,100],
                            [100,1,1,100,1,1,1,1],
                            [100,1,1,100,1,1,1,1],
                            [1,100,100,1,100,100,1,1],
                            [1,1,1,100,1,1,100,1],
                            [1,1,1,100,1,1,100,1],
                            [1,1,1,1,100,100,1,100],
                            [100,1,1,1,1,1,100,1]]
        while True:
            self.re_num = 0
            lens=len(self.datapaths)
            self.link_bw_dict=np.array(self.link_bw_dict_o)
            self.link_bw_dict[self.link_bw_dict>1]*=2**10
            self.traffic=self.link_bw_dict
            self.traffic[self.traffic==1]=-1
            # self.traffic = np.ones((lens*lens),dtype=float) *  -1
            self.traffic=np.array(self.traffic)
            self.traffic=self.traffic.reshape((lens*lens))

            for dp in self.datapaths.values():
                
                self._request_stats(dp)

            while self.re_num != lens:
                hub.sleep(0.5)
            
            a = self.traffic.reshape((lens,lens))

            free_bw = np.ones((lens,lens),dtype=float) *  -1
            if self.weight_length==5:
                slice_input=np.array(self.flows)
                #print(self.flow_input)
                #print(slice_input)
                for i in range(20):
                    #print('flow_input:',slice_input[:,0])
                    self.weight_sum.append(np.sum(slice_input[:,i].reshape(1,self.weight_length))/self.weight_length/1024)
                
                print('self.weight_avg:',self.weight_sum)
                freebw_var=np.var(self.weight_sum)
                print('Link variance:',freebw_var)
                print('================================================')
                df = pd.DataFrame(np.expand_dims(list(freebw_var.reshape((1*1))), axis=0))
                df.to_csv("~/ryu/ryu/app/GCN/var2_8.csv",mode='a',header=False,index=False)
                self.flows=[]
                self.weight_sum=[]
                self.weight_length=0
            weight_input = []
            for i in range(lens):
                for j in range(lens):
                    if not self.graph.has_edge(i+1,j+1):
                        continue
                    temp=self.link_bw_dict[i][j]-a[i][j]+1
                    if temp <0:
                        temp=0
                    free_bw[i][j] = temp
                    weight_input.append(temp)
            
            self.flows.append(weight_input)
            self.weight_length+=1
            print('self.weight_length:',self.weight_length)
            


            #df = pd.DataFrame(np.expand_dims(list(free_bw.reshape((lens*lens))), axis=0))
            #df.to_csv("~/ryu/ryu/app/GCN/flow17.csv",mode='a',header=False,index=False)
            #print('writ end')
            freebw_max=np.max(free_bw)
            freebw_mean=np.mean(free_bw[free_bw>0])
            
            print('free_bw',free_bw[free_bw>0])
            
            print('Link average residual bandwidth:',freebw_mean)
            #freebw_var=np.var(free_bw[free_bw>0])
            #print('Link variance:',freebw_var)
            #df = pd.DataFrame(np.expand_dims(list(freebw_var.reshape((1*1))), axis=0))
            #df.to_csv("~/ryu/ryu/app/GCN/var2_2.csv",mode='a',header=False,index=False)
            link_performance=free_bw[free_bw>-1]/self.link_bw_dict[self.link_bw_dict>1]
            link_per_mean=np.mean(link_performance)
            # print('Link performance:',link_per_mean,link_performance)


            self.avg_rest.append(freebw_mean)
            freebw_log=np.log10(free_bw)/np.log10(freebw_max)
            freebw_weight=1-freebw_log
            #print(freebw_weight)
            for i in range(lens):
                for j in range(lens):
                    if not self.graph.has_edge(i+1,j+1):
                        continue
                    self.graph[i+1][j+1]["weight"]=freebw_weight[i][j]
            # print(self.graph.edges(data=True))
            #hub.sleep(10


    def _request_stats(self, datapath):

        """

            Sending request msg to datapath

            发送请求端口信息到数据路径（获取交换机端口信息）

        """

        self.logger.debug('send stats request: %016x', datapath.id)

        ofproto = datapath.ofproto

        parser = datapath.ofproto_parser

        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)

        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):

        """

            Save port's stats info

            Calculate port's speed and save it.

            保存端口的统计信息

            计算端口的速度并保存。

        """
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        for stat in sorted(body, key=attrgetter('port_no')):

            port_no = stat.port_no

            if port_no != ofproto_v1_3.OFPP_LOCAL:

                key = (dpid, port_no)

                value = (stat.tx_bytes, stat.rx_bytes, stat.rx_errors,

                         stat.duration_sec, stat.duration_nsec)

                self._save_stats(self.port_stats, key, value, 2)                # 将key(交换机，端口) 和 value(接收数据，发送数据，接收错误，时间1，时间2) 添加到 self.port_stats

                # Get port speed. 获取端口流速

                pre = 0

                period = 0

                tmp = self.port_stats[key]

                if len(tmp) > 1:

                    # 传输字节和接收字节相加

                    pre = tmp[-2][0] + tmp[-2][1]

                    # 计算间隔时间

                    period = self._get_period(tmp[-1][3], tmp[-1][4],

                                              tmp[-2][3], tmp[-2][4])

                # 计算端口速度,总字节数减去上一时间段的总字节数除于间隔时间

                speed = self._get_speed(

                    self.port_stats[key][-1][0] + self.port_stats[key][-1][1],

                    pre, period)

                # 将端口速度保存(dp.id,port_no) -> speed

                # print('speed:',speed)
                # print('dpid_port:',self.dpid_port)

                if key in self.dpid_port:

                    self.traffic[(dpid-1) * len(self.datapaths) + self.dpid_port[key][0]-1] = speed                                      # 流量矩阵
        self.re_num = self.re_num + 1

    def _save_stats(self, _dict, key, value, length):

        "将信息保存到 _dict[key] = value 最大长度为length，超过弹出第一个"

        if key not in _dict:

            _dict[key] = []

        _dict[key].append(value)

        if len(_dict[key]) > length:

            _dict[key].pop(0)

    def _get_period(self, n_sec, n_nsec, p_sec, p_nsec):

        " 计算间隔时间"

        return self._get_time(n_sec, n_nsec) - self._get_time(p_sec, p_nsec)

    def _get_time(self, sec, nsec):

        # 计算持续时间

        return sec + nsec / (10 ** 9)

    def _get_speed(self, now, pre, period):

        # 计算带宽

        if period:

            return (now - pre)*8/ (period) /2**10

        else:

            return 0

    def _get_free_bw(self, capacity, speed):

        # BW:Mbit/s

        return max(capacity/10**3 - speed * 8/10**6, 0)



    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)

    def packet_in_handler(self, ev):

        "Packet_in 包入操作"
    # print(self.graph.edges(data=True))

        msg = ev.msg

        dp = msg.datapath

        ofp = dp.ofproto

        parser = dp.ofproto_parser

        pkt = packet.Packet(msg.data)      

        eth_pkt = pkt.get_protocol(ethernet.ethernet)

        dst = eth_pkt.dst

        src = eth_pkt.src

        dpid = dp.id

        in_port = msg.match['in_port']

        if dpid not in self.datapath_switch:

            self.datapath_switch[dpid] = dp

        if eth_pkt.ethertype == ether_types.ETH_TYPE_LLDP:

            return

        if eth_pkt.ethertype == ether_types.ETH_TYPE_IPV6:

            return

        header_list = dict((p.protocol_name, p) for p in pkt.protocols if type(p) != str and type(p) != bytes )

        # 判断是否是ARP且dst_mac还未学习

        if dst == ETHERNET_MULTICAST and ARP in header_list:

            # 记录源str的IP和MAC

            self.arp_table[header_list[ARP].src_ip] = src

            # 获取dst的ip

            arp_dst_ip = header_list[ARP].dst_ip

            # 判断dst是否有对应mac,没有就像除本交换机外的所有交换机连接主机的端口发送包

            if arp_dst_ip not in self.arp_table:

                for key in self.switch_host_port:

                    # 交换机id

                    if key != dpid:

                        dp = self.datapath_switch[key]

                        for out_port in self.switch_host_port[key]:

                            out = parser.OFPPacketOut(

                                datapath=dp,

                                buffer_id=ofp.OFP_NO_BUFFER,

                                in_port=ofp.OFPP_CONTROLLER,

                                actions=[parser.OFPActionOutput(out_port)], data=msg.data)

                            dp.send_msg(out)
                            return

            else:

                dst = self.arp_table[arp_dst_ip]

        # 最短路径寻找

        self.mac_table.setdefault(dpid, {})

        if dst  in self.mac_table[dpid]:

            out_port = self.mac_table[dpid][dst]

        else:

            out_port = ofp.OFPP_FLOOD

        # 源主机未在图中需添加，且权值为0

        if src not in self.graph:

            self.graph.add_node(src)

            self.graph.add_edge(dpid, src, weight=0, port=in_port)

            self.graph.add_edge(src, dpid, weight=0)
	
        if src in self.graph and dst in self.graph and dpid in self.graph:

            # 找出最短路径，跳数最短

            path = nx.dijkstra_path(self.graph, src, dst, weight="weight")


            print(path)				#实验一打印路径
            if dpid not in path:

                return

            # 找出后一跳的交换机

            nxt = path[path.index(dpid) + 1]

            # 找出应该发送的端口

            out_port = self.graph[dpid][nxt]['port']

            # 记录交换机发往目标地址应该发送的端口

            self.mac_table[dpid][dst] = out_port

            actions = [parser.OFPActionOutput(out_port)]

            out = parser.OFPPacketOut(

                datapath=dp, buffer_id=ofp.OFP_NO_BUFFER, in_port=in_port, actions=actions, data=msg.data)

            dp.send_msg(out)

            # 下发流表

            match =  parser.OFPMatch(in_port=in_port,eth_dst=dst,eth_src=src)


            self.add_flow(dp,1,match,actions,5)	#实验二定时删除流表
            # self.add_flow(dp,1,match,actions)

        else:

            actions = [parser.OFPActionOutput(out_port)]

            out = parser.OFPPacketOut(

                datapath=dp, buffer_id=ofp.OFP_NO_BUFFER, in_port=in_port, actions=actions, data=msg.data)

            dp.send_msg(out)


 
