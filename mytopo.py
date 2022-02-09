#!/usr/bin/python
 
from mininet.topo import Topo
 
class MyTopo( Topo ):
 
    def __init__( self ):
 
       "Create custom topo."
 
       # Initialize topology
       Topo.__init__( self )
 
       # 生成所需要的主机和交换机
       h1 = self.addHost( 'h1' )
       h4 = self.addHost( 'h4' )
       h7 = self.addHost( 'h7' )

       sw1= self.addSwitch( 'sw1' )
       sw2 = self.addSwitch( 'sw2' )
       sw3= self.addSwitch( 'sw3' )
       sw4= self.addSwitch( 'sw4' )
       sw5= self.addSwitch( 'sw5' )
       sw6= self.addSwitch( 'sw6' )
       sw7= self.addSwitch( 'sw7' )
       sw8= self.addSwitch( 'sw8' )


 
 
       # 添加连线，交换机和交换机之间，交换机和主机之间
       self.addLink( sw1, sw2, bw=300, delay='5ms')
       self.addLink( sw1, sw3, bw=200, delay='5ms')
       self.addLink( sw2, sw4, bw=100, delay='5ms')
       self.addLink( sw3, sw4, bw=300, delay='5ms')
       self.addLink( sw4, sw6, bw=300, delay='5ms')
       self.addLink( sw4, sw5, bw=100, delay='5ms')
       self.addLink( sw5, sw7, bw=300, delay='5ms')
       self.addLink( sw6, sw7, bw=200, delay='5ms')
       self.addLink( sw1, sw8, bw=100, delay='5ms')
       self.addLink( sw8, sw7, bw=100, delay='5ms')
       self.addLink( sw1, h1)
       self.addLink( sw4, h4)
       self.addLink( sw7, h7)

#实例化类
topos = { 'mytopo': ( lambda: MyTopo() ) }

