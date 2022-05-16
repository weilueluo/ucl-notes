# 	Network Systems COMP0023

### What We Will Learn

- Architecture: IP / TCP / DNS / BGP.
- Lower level: Ethernet / wireless / error correcting codes.
- Higher level: the Web / other applications.
- Components Technologies: switches / routers / firewalls.

### Challenges

> - Physical Limitations: finite speed of lights & scare bandwidth
> - Challenges: re source sharing & scalability & reliability
> 
> Implications: asynchronous communication design, distributed coordination, failures as commonplace & flexibility...

- Physical limitation: speed of light and bandwidth
  
  - home: ~50ns.
  - local: ~1$\mu$s.
  - country: ~50ms
  - satellite: ~0.2s
  
  *Round-trip-time will double the above, and further double in practical real-life scenario.

- All network must be shared
  
  - switch: a device that allows many users to share a network.

- Failures are common case
  
  - cables can be cut.
  - wireless signals contend to hostile radio environment (scattering, diffusion and reflection).
  - any network component may (and eventually) break.

- Incommensurate 不可估量 growth.

- Large Dynamic Range (of parameters)
  
  - round-trip-time varies from eight order of magnitudes (ns to sec).
  - data rates varies from kb/s to gb/s.
  - bit error varies from $10^{-8}$ to $10^{-1}$.
  - packet loss varies from $10^{-6}$ to $10^{-1}$.

- Diversity
  
  - applications requirements
    - size of transfers
    - bidirectionality
    - tolerance for latency / packet loss /  jitter.
    - need for reliable & multi-point communication.
  - end systems ranged from mobile phones to supercomputers.

## Internet Service Model

### Shared Network

If we want $N$ hosts to communicates, we need $N^2$ dedicated links. Deploying such amount of links are not possible, partially because the cost of deploying and maintaining network links are high, and requires approval from regulatory (glacial pace). Therefore there are strong economic motivation to share resources & links.

Telephone Network: In the early days, people manually link telephone lines, now it is automated but the service model did not change. The company utilize bandwidth in following ways:

**Time-division multiplexing**: divide time in regular intervals to form frames, each telephone communication uses 1 frame for every $n$ intervals (by some quality-of-service standard), and $n$ will be the maximum amount of communication it can serve. In this model, if the user get the service, there will be predictable rate and delay because they get equal-sized frames at equal intervals.
**This model does not fit the internet** because users in the network send data in burst instead of steady stream as in telephone calls. (Frequency-division multiplexing has the same limitation). 

A better model would be send data as it becomes available without schedule and transmit asynchronously. We can support burst and remove the user limit, but we are giving up predictability & latency. Also statistically, the users will not send at the same time, so the network is smooth overall (This is called **statistical multiplexing gain**).

To send data asynchronously, hosts divide data into datagrams (generally called packets) with guidance information on how to send it as header. These packets can be of any length (required framing) and send anytime local link is free; this allows the network to be stateless. This means that there is no start up cost to setup connection, allowing flexibility, scalability & resilience, e.g. packets can be re-routed on-the-fly if something goes down. This is one of the four internet design principles: datagram packet forwarding.

### Datagram-based Forwarding

In order for a packet to go from $A$ to $B$, it has to go through a number of intermediate systems, (like switches / routers), they first examine guidance information in the packet; consult their forwarding table and forward the packet to the destination. 

But how does the system knows which route to take? They use something called forwarding table, explained in later weeks, the building of the table is called routing.

Factors affecting the transmit time:

- Propagation delay: speed of light over the medium, fixed.

- Transmission delay: depends on the quality of links, varies.

- Processing delay: checksum, copying, forwarding decision, fixed.

- Queuing delay: depends on the amount of traffic, varies.
  
  - The average queuing time according to queuing theory: $1/(1-p)$ where $p$ is the utilization of outgoing link. The link providers want high utilization because maintaining links cost money. User wants to have low latency, so there is usually a maximum tolerable delay for the user and this is also the maximum utilization ($p_{max}$) the link provider can get.
  
  The trade-off between delay and utilization take place through out computer systems. But note that the queuing theory assumes 

- packets arrive according to random & memory-less process

- packets have randomly distributed service time (i.e. transmission delay)

- utilization of outgoing link is $p$.

which does not hold on real data network, packet arrivals are bustier (instead of random) and the queue in router is finite, they have some upper bound for delay. But how much memory does each router needs? (dimension switches' queues). Memory is relative cheap so can we use the worst-case memory size? No, because worst-case is orders of magnitude larger than average, and even if we can queue everything, the user will rather retry or just give up than wait for long time.

So we use average sized memory, but what to do when it is full (congestion)? We cannot ask the sender to slow down (send a quench message) because that will generate more traffic and the source may not be sending anything more. So when the queue is full, we just simply drop the packet, and sender will need to resend the packet (if he cares). This implicitly tells the sender that the traffic is congested now (possibility of slowing sender automatically to response to it), and this is what the internet does.

### Best Effort Delivery

Networks that never discard packets are called **guaranteed-delivery**, which potential for higher delay, and network such as the internet is called **best-effort**. Note that internet applications are required to build guaranteed delivery on-top-of best-effort delivery (e.g. email, trying again and again).

Implications:

- Packet loss.
- Duplication: duplicate send because sender did not receive a respond.
- Unbounded Delay: receiver sent a response but was delayed, so sender send again and received two responses as well, also causes duplication.
- Noise on links: packet can be corrupted, but error correction is mostly efficient and will drop these packets.
- Links break: packet can be rerouted when failure.
- Out-of-order delivery: receiver need to reassemble packets.

### Summary

<img title="" src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/02/upgit_20220214_1644877070.png" alt="image-20220120215434366"  width="399">

## Internet Protocol Stack

- Protocol: an agreement on how to communicate between two parties.
  - syntax: how communication is specified and structured (e.g. format / order of the messages sent)
  - semantics: what communication means, e.g. actions are taken during events (e.g. transmitting / receiving / time-out)

In order for applications to communicate with each other, they have to transmit message, there many means of transmission (e.g. fiber optic 光纤, coaxial cable 同轴电缆 and wifi) and we cannot expect every application to implement all means of transmission, therefore we partition the them into modules / abstractions (i.e. layers) where each layer relies on layers below and export services to layer above. Now each layer only needs to implement intermediate layer's interface.

Advantages:

- break down problem into manageable pieces.
- isolate changes.
- hide complexity.

Disadvantages:

- headers are overhead. (e.g. header is bigger than content)
- same information maybe duplicated (needed) at each layer. (e.g. timestamp)
- layering can hurt performance. (hiding details)
- same functionality maybe duplicated at each layer (error recovery)

The stack:

- Application
  - email, www, phone
  - SMTP HTTP RTP
- Intermediate layers
  - TCP UDP
  - IP
  - ethernet PPP
  - CSMA async sonet
- Transmission media
  - copper fiber radio

Layers:

1. **Physical**: move bits between two system connect by single physical link
2. **Data Link**: packet (L2 frames) exchange between hosts in the same LAN using abstract address (e.g. MAC: CSMA/CD), may implement some form of reliability.
3. **Network**: deliver packets (varying frames) to destination on other local networks (inter-networking) using globally unique addresses. Include packet scheduling / buffer management.
4. **Transport**: end-to-end communication (bytes stream) between processes on different hosts (demultiplexing). possibly implement reliability (retransmission / duplicate removal) and rate adaptation (congestion / flow control)
5. **Session**
6. **Representation**
7. **Application**: any services provided to end-user, application dependent interface & protocol, e.g. FTP / Skype / SMTP / HTTP/ BitTorrent...

Session and Representation layers are part of **OSI architecture** but not **internet architecture**; and transport and application layer are implemented only at end hosts.

When sending, layer adds its own header (encapsulate), which then get inspected and removed when received (decapsulate).

Each underlying protocol also keep track of the overlying protocol, so that they know where should the data be send to.

### The end-to-end Principle

So where do we implement the following functionalities:

- packet corruption or loss
- out-of-order delivery
- duplicates
- encryption
- authentication

## Reliability

1. No corrupted packet.
2. All submitted packets are delivered.
3. Packets are in order of submission.

### Physical Layer

We need to design and implement reliable packet delivery over unreliable physical layer. We can measure reliability using **Bit Error Rate (BER)**, in real life it is around $10^{-6}\sim 10^{-8}$,note that it assumes bit errors are independent random uniformly distributed which is not true, real bit errors are unpredictable and bursty.

If we assume each frame contains $12000$ bits and BER is $10^{-6}$, then **Frame Error Rate (FER)** is $1-(1-10^{-6})^{12000}=1.19\%$ which is relatively high. We can either:

- Reduce number of bits per frame which will leads to higher header overhead.
- Reduce BER by introducing **Error Control Coding (ECC)** $\rightarrow$ not internet specific.
  1. Detect error by computing same code (e.g. CRC, parity).
  2. Respond: correct using error correction code / sender resend errored frame (receiver sliently discard errored frame, let the higher level to handle it).

Simple ECC $\rightarrow$ hamming distance repetition, we can correct $\frac{d_{min}-1}{2}$ bits. We can measure its **code rate**: $\frac{\text{len(message)}}{\text{len(message+code)}}$ $\rightarrow$ trade-off: high code rate means correct less error, low code rate means higher overhead.

Parity Check $\rightarrow$ 1 if the number of 1s in the message is odd. The hamming distance is 1 for 1d, because we need to flip at least 2 bits to receive a corrupted message; if it is 2d then we need to at least change 4 bits to receive a corrupted message, so $d_{min}=4$. In general, code words with length $n$ that is parity-encoded message of length $k$ is called $(n,k)$ message block, commonly used is $(7,4)$, its has $d_{min}=3$, can detect 2 bits errors and correct 1 bit error.

### Link Layer

Although in practice ECC in physical layer can handle many errors, some errors can still slip through, therefore we need a error detect scheme that ensure a low probability of missing error. In this case, the **Cyclic Redundancy Check (CRC)** is a popular choice to quickly filter out corrupted frames (often hardware implementation).

CRC represents $k$ bits message as $k-1$ polynomials, where each coefficient is either $0$ or $1$. It uses modulo-2 arithmetic, where both addition and subtraction are exclusive-or without carry/borrow, the algorithm goes as follows:

1. Sender and receiver agrees on a generator $G(x)$ of degree $g-1$.
2. Sender
   1. $\text{pad\_msg} = message + g-1(zeros)$.
   2. $\text{reminder}= \frac{\text{pad\_msg}}{G(x)}$.
   3. $\text{transmit codeword}= \text{pad\_msg} + \text{reminder}$.
3. Receiver
   1. reminder = codeword / $G(x)$.
   2. if (reminder != 0) $\rightarrow$ error occur
   3. else $\rightarrow$ maybe no error

CRC can:

- Detect all single bit errors **IF** $G(x)$ has at least two non-zero terms.
- Detect all burst errors of length less than $g-1$ **IF** $G(x)$ begin and ends with $1$.
- All odd number bit error **IF** $G(x)$ contains even number of ones.

TODO: CRC

### L3/L4 Internet Checksum

Most errors are already picked up, but this is the last layer that can detect error before reaching the application; it is simple and fast, implemented in software, it is the only checksum that take place end-to-end (transport layer).

TODO: ones complement

Break packets into segments of 16 bits and sum them using ones complement, then negate the result and add the header.

- space-efficient but only detect 1 bit error.

### Reliable Delivery

#### Stop-and-wait

So far we have FEC in physical layer that can correct some errors, and error detection in higher layers, such as CRC in link layer and IC in network layer. But this still does not ensure reliability.

We could put less bits in each frame to reduce Frame Loss Rate (FLR) or adds more FEC, but they also introduce overheads which waste a lot of capacity. Thus, detect and discard error frame is the only option, and most errors can be detected by the 32-bits CRC and IP checksum,

So, in practice, we:

- Ask for retransmit when error is detected

- Add just enough FEC to keep FLR below the knee (typically FLR < $10^{-3}$).

Base on retransmit, let's develop a scheme by slowly relaxing the following three assumptions:

1. Channel does not corrupt packets
2. Channel always delivers packets
3. Channel does not delay or reorder packets
- First of all, if all three assumptions holds, we just send messages one-by-one, done.

- Now if $1$ does not hold:

  - We may try to send ACK/NACK message to the sender to acknowledge them if the message has been corrupted, but this ACK/NACK message can also be corrupted, so it does not work.

  - Another way is to always resend when in doubt (doubt=NACK received), but now we may receive duplicates if ACK is corrupted.

    - We can introduce a 1-bit sequence number to resolve this, if we receive the same sequence bit message, that means we have duplicates so we just discard it. We can also drop NACK message here and use this sequence bit to tell the sender if we have receive the last packet or not (e.g. if we expect sequence 0 but did not receive it, then we send 1 back to the receiver, otherwise 0).
    
      - > TODO: If the bit sequence also corrupted, then there is nothing we can do? Rarely happen I suppose

- Now if $2$ does not hold:

  - add a timeout, resend when doubt, (now doubt = timeout or NACK received).

- Now if $3$ does not hold:

  - Now if one packet is delay, it will cause endless retransmission (when the first ACK reach before after timeout on sender).

  - We can redesign, and use a **stop-and-wait** method:

    1. You send a packet, and set timer to expect that sequence number
    2. ignores any ACK that is not the expecting sequence number
    3. Wait until timeout or ACK of that sequence number received
    4. send next packet.

    <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/02/upgit_20220214_1644878912.png" alt="image-20220214224830410" style="zoom:50%;" />

    Note that in practice, people use larger sequence number, because sender may give up on sending a packet if it keeps failing; and if sender skips a packet, a 1 bit sequence number will cause the receiver to think it is a duplicate.

In term of utilization, it is: $U=\frac{L\,\cdot\, RRT}{R}$, where $L$ is the number of bits to send, $R$ is the bandwidth (number of bits transmit per seconds), and $RRT$ is the round-trip time. In traditional wifi link, like $L=8000, R=54Mbps,RTT=100ns$ then we have $U=99.93\%$. But if we sending them across the internet, we will have lower $R=10Mbps$ and $RTT=30ms$, which will result in $U=2.1\%$ which is very poor.

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/02/upgit_20220215_1644932919.png" alt="image-20220215134838948" style="zoom:50%;" />

In this setup, notice that the sender and receiver spends most of the time waiting for the message, this is because the bottleneck is no longer the speed of sending / receiving, but the time taken for the data to travel across the internet. So in practice, we send multiple packets in one go:

- 802.11n: do not send more than 4ms in one go and up to 64kb (64 packets in one go), this increases performance up to hundreds of Mbps and link utilization.

For the receiver, he need to send one block ACK that tells the sender what are the packets has been received. So in summary **stop-and-wait**:

- requires receiver needs to be able to buffer 64 packets.
  - Note if one packet is dropped, the sender can only resend that packet in the next time because there is no space in the buffer to put more packets.
- block ACK is 64-bits vector where $i$-th bit indicates whether that packet is received.
  - Now this block ACK is critical because if it is corrupted/missing, sender needs to resend all the packets after timeout.
    - We can request block ACK again after timeout.
- is still slow for long RTT, because even if we increase the size for one go, we still need to wait for the block ACK from receiver

#### Go Back N (GBN)

We allow the receiver to send ACK asynchronously, instead of wait for everything (? is this true?). Now we achieve high utilization.

Sender use a sliding window approach and needs to know the last packet that is correctly received in order.

So the receiver need to send ACK to sender with the number that the last packet was received in order. Let say we are sending packet 1,2,3,4

- If only packet 2 is lost while sending, then we send ACK 1,1,1, because the the last packet that received in order is 1, and the sender will know 1 is correctly received, followed by 2 other packets, but he does not know which one is lost.
- If only packet 2's ACK is lost while sending, then we receive ACK 1,3,4, and the receiver will know all packets have been received correctly, because when he received ACK 3, he knows ACK 2 is lost but successfully transmitted.
- If packet 2 is lost and we received ACK 1 before sending packet 4 and we received another ACK 1, then we know that packet 4 has been successfully transmitted (? what if out-of-order but none lost?)

The sender will need to resend all packets after the last packet that he was acknowledge (timeout / ACK from receiver), hence go back n.

- a timeout timer is started when a packet is sent, and canceled when its ACK is received.
- when a timer expires, go back to it and resend from there.

We can improve efficiency using **fast retransmit**

- duplicate ACKs are sign of packet loss / network reordering
  - retransmit after a few duplicates (empirically, 3 is used ).
- we can use **selective ACK (SACK)**, append to header specifying which packets it have received but it is not in order, e.g. if we send 1,2,3,4 and 2 is loss, then we send back ACK 1; ACK 1, SACK 3; and ACK 1, SACK 3,4, but this also **increase header size**.

#### Utilization

$$
\begin{align*}
U&=\frac{N \times \frac{L}{R}}{RTT}\\
&=\frac{N\times L}{RTT\times R}
\end{align*}
$$

when $U=1$, we have $N\times L=RTT\times R$, which means number of bits in flight is equal to the bandwidth-delay product. We cannot achieve full utilization in a network because different links have different bitrate.

## TCP

So now we can transport packet reliably through the best-effort network layer that handles drops, delays, reorders and corrupts packets. But many applications wants reliable transport that ensures all data reaches the receiver in order, specifically:

- connection-oriented (not entirely secure)
  - uniquely identified by sender ip & port, receiver ip & ports and protocol.
- reliability
  - recovery (data loss)
  - duplicate
  - order
  - integrity (corruption)
- transfer as fast as possible
  - avoid send faster than receive
  - avoid congesting network

Application listen to port, and os send the received data to corresponding port. Well known ports include TCP-HTTP:80, TCP-SMTP:25, TCP-SSH:25, UDP:53, etc... and each port can only be owned by one application instance.

Problems:

- how sender and receiver agrees?
- data from old connection received?
- prevent host impersonation.

### At-least-once Delivery

We have seen that we can attach unique nonce (sequence number) to packet and wait for its ACK to ensure the packet is actually delivered, there are a number of problems unaddressed:

#### Number of retransmissions

Set by sender.

#### Duration of Timeout

Too short leads to frequent retransmissions and too long leads to delay in detecting loss. We cannot set a fixed value because this apply assumption about the underlying link layer (congestion / route changes). We can tune the timeout using RTT from previous packets.

- Adapt over time, using **Exponentially Weighted Moving Average (EWMA)**
  - $RTT_i=\alpha RTT_{i-1} + (1-\alpha) m_i$ where $m$ is the measurement for current delivery. We choose $\alpha$ as $0.9$ (empirically).
  - Original TCP: $\text{timeout}_i=\beta\times RTT_i$ where $\beta=2$.

#### Size of Nonce Space

- If we use random number then receiver has to store all of them in order to verify whether we have seen the current packet before (we need to drop duplicate!)
  - We **use cumulative sequence number**, increase monotonically, receiver can drop packets that has been received in order. Although this reduces the problem, we still need to set the sequence space for the number, we come back to this later

#### Preserve data ordering

- transport layer **segment** the data and **reassemble** at the receiver side, **mark each packet by its range of bytes in original data**, pass to receiver when all bytes before some point has been received.

#### End-to-end integrity

- Use **internet checksum over**:
  - **payload** protect against link layer reliability.
  - **transport protocol header** protect header sequence number and payload mismatch.
  - **layer-3 source and destination** protect against delivery to wrong destination.
- cannot protect against software bugs / router memory corruption, etc...
- drop when checksum fails.

### Performance

Do not use stop-and-wait because it is too slow --- we need to wait for ACK every time we send a packet. For example, if we have 70ms RTT, then 1500-byte packets can only delivery up to 171 kbps.

Instead we pipeline transmission by using sliding window: Go-back-n, fast retransmit, SACKs.

#### Sliding Window Size

From utilization point of view, it should be at least equals to the bandwidth-delay product so that we can achieve maximum utilization, but this is not always optimal, because:

- Can receiver sustain such rate? (flow control) 
- What if the bottleneck link is shared? (the bandwidth-delay product will be smaller this case) Can the network cope with such rate? (avoid congestion)

### Header

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/02/upgit_20220218_1645221579.png" alt="image-20220218215938339" style="zoom: 67%;" />

- Min 20 bytes
- Checksum includes tcp segment (payload) + pseudo header (sender and receiver address, protocol and tcp segment length).
- 16 bits port (0-65535)
- bidirectional (carry sequence number for both data and ACKs)
- 32 bits sequence number
  - as packet id and reassembly
  - enough to avoid wrapping issue (so far)
- window: number of bytes advertiser willing to accept in addition to bytes ACKed in the packet. (avoid overwhelming the receiver)
- Flags
  - SYN, exchanged to establish connections
  - RST, reset, forget about the last connection
  - FIN, finish, close connection

### Idea

- Start TCP connections between two hosts
- Avoid mixing packets between connections
- Avoid confusing connection attempts
- Prevent impersonation



- connection cannot starts with constant sequence number, because it mixes data between old and new connections (? what if it is just loss packets at the beginning)
- need to use random starting sequence number and explicitly ACK with the received sequence number; if unrecognized sequence number received, response with RST reset.



- 3-way handshake
  - send own sequence number to each other and confirm we received the corresponding ACK
  - hard to impersonate because the attacker does not know the sequence number send, so he cannot send the corresponding ACK
- each side independently decide when to close, need to agree last data are sent and connection is ended
  - to avoid mixing new and old connection data, when connection tear down, one endpoint enters TIME_WAIT for a long enough time (2x max segment life time), disallow new connections.

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/02/upgit_20220219_1645272680.png" alt="image-20220219121119496" style="zoom: 80%;" />

### Data Transmission

- Sender use a sliding window that does not exceed send / receive window size and the network estimated capacity with timeout
- receiver send cumulative ACKs
  - ACK number is the highest contiguous byte number so far + 1.
  - there are also delayed ACKs that batched into one for every pair (max delay 200ms)

#### Utilization

- Sender just send all packets advertised by the receiver and retransmit after timeout, it is a pure go-back-n.
  - window-sized bursts of packets sent into network at max bit rate

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/02/upgit_20220219_1645280622.png" alt="image-20220219142340683" style="zoom:50%;" />

- **transmission rate** burst leads to congestion collapse.
- **transmission timeout** due to larger RTT due to queuing at intermediate link.
  - leads to timeout & retransmission.

We was using Exponentially Weighted Moving Average to estimate RTT, and timeout = 2 * RTT. But this clearly does not work because when we transmit in burst, increase in timeout will cause this timeout to happen too early. So Jacobson proposed to use **Retransmission Timeout (RTO)**:

- estimate $v_i=\text{mean deviation}=|m_i-RTT_i|$ to approximate RTT variance
- $RTO_i=RTT_i+4v_i$.

Next we solve what is the max rate to send to achieve max utilization?

- what is the initial rate
- how to adjust the rate

We starts with 1 then increase by packet size (double) until receiver advertised window size (slow start). Takes $\log_2(\text{receiver window size} / \text{packet size})$ RTTs to reach receiver window size. Now if we use this slow start and mean+variance RTT estimator, we have:

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/02/upgit_20220219_1645285080.png" alt="image-20220219153759790" style="zoom: 67%;" />

which is pretty close to the optimal bandwidth compared to the original TCP, and we have almost no retransmission at all (we did not overload the network).

#### Congestion Control

Below shows congestion collapse

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/02/upgit_20220219_1645286805.png" alt="image-20220219160644082" style="zoom:50%;" />

To avoid congestion collapse the only way is to slow down.

- Absence of ACK implicitly indicates that the network is congested.
  - It can also be corrupted, but most of the absence is due to network congestion.
  - ACKs return: window ok.
    - We increase the window size by packet size: $\text{cwnd}=\text{cwnd}+(\text{pktSize}\times \text{pktSize}) / \text{cwnd}$. (increase 1 each RTT)
  - ACKs missing: window too big.
    - We decrease the window size, but practically, when we have missing ACKs, the network maybe just not working at all, so instead of decreasing it by 2, we set the slow start threshold (ssthresh) to current window size / 2 and slow start from 0 up to that value.

This is called **Additive Increase, Multiplicative Decrease (AIMD)**.

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/02/upgit_20220221_1645465683.png" alt="image-20220221174802256" style="zoom: 67%;" />

We can visualize the efficiency and fairness between two hosts using Chiu-Jain Phase Plot:

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/02/upgit_20220221_1645465924.png" alt="image-20220221175203992" style="zoom:67%;" />

For AIMD we have:

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/02/upgit_20220221_1645466040.png" alt="image-20220221175400759" style="zoom:50%;" />

#### Performance Analysis (++)

- Assume window size over time looks like: <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220513_1652464753.png" alt="image-20220513185911889" style="zoom:33%;" />, then mean window size = $3W/4$.
- mean throughput = mean window size * bits per RTT / RTT = $3BW/4RTT$.
- we lost one packet per cycle, number of packets we send each cycle  = packet/time * total time = $(3W/4)/RTT \times (W RTT / 2) = \frac{3W^2}{8}$.
- so loss rate is 1 / packets in a cycle = $p=\frac{8}{3W^2}$.
- so when we lost one packet every $W=\sqrt{\frac{8}{3p}}$ window size.
- substitute into mean throughput: <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220513_1652466001.png" alt="image-20220513192001783" style="zoom:33%;" />.



- Higher RTT, loss reduce throughput
- At same bottleneck, flow with longer RTT achieves less throughput than flow with shorter RTT!

## L2 Forwarding

How do hosts delivery packets?

- Forward the packet to next (forwarding).
- Host bootstrapping (ARP, DHCP).
- Best path selection (routing).

### Data Link Layer

Enable packet delivery within the same Local Area Network (LAN)

- Abstract addresses.
- Multiple physical links.
- Access to shared media.
- Some form of reliability.

Uses **Media Access Control (MAC)** such as CSMA/CD.

- Identifier in the link layer (within the LAN).
- Assigned by **Network Interface Card (NIC)** vendors.
- **48 bits** (6 bytes, first 3 bytes identify the vendor, globally unique).
- burned in NIC ROM, sometimes software settable, flat address space (unstructured).

The address to passed to higher level if:

- It is **broadcast address** $\text{ff:ff:ff:ff:ff:ff}$.
- It is in **promiscuous mode** (catches all frames and pass to higher level).
- It is NIC address.

#### Repeaters

**Repeaters** rebroadcast all bits received in the physical-layer frames.

- Used to join cables and amplify signals to avoid weaken over time.
- **Hubs** are repeaters that can join multiples cables (but may not amplify signals).

Problems:

- Does not interpret signals so cannot interconnect different formats/rates.
- Does not avoid collision. (use MAC to resolve this)
- Max node/distance are as on single-cable LAN (e.g. <2500m in commercial ethernet).
- Hosts connected via a cable will share its maximum speed (limit throughput).

#### Switches

**Switches** forward link-layer frames based on link-layer header.

- Connects different collision domains (each port define different domain).
- Extract link address and forward selectively to their collision domain.
- More and more hosts are connecting to switches directly.

Advantages:

- Full duplex (send and receive bidirectional).
- No collision -> no carrier sense collision detection, change in medium access control but same framing.
- Change in MAC but same framing. (change l2 header but same framing)
- Can combine different technologies since it can re-encapsulate packets.
- Avoid unnecessary load on connected LAN segments (when it builds the spanning tree, unlike hub)
- Improves privacy as each switch can only snoop at their own traffic.

Most LANs are designed like this today.

Disadvantages:

- It has higher cost and introduce delay
  - need to parse and decide what to do. (**store-and-forward delay**)
    - Can be ameliorated 改善 by **cut-through switching**
      Start parsing when we have received the header instead of waiting for the whole packet.



#### Routers

- **Routers** forward IP datagrams based on network-layer header.


### Forwarding Table

Switches maintain a forwarding table that maps an destination address to a outgoing port, we would like to construct such table automatically without intervention from administrator via a self-learning algorithms.

Let's start with an simple idea: starts a empty table, when we receive any request sending from $A$ to $B$ from a port $P$:

- We add $A$ and $P$ in the forwarding table because we know port $P$ leads to address $A$; also a time-to-live since the network may change.
- If $B$ is in the forwarding table and time-to-live is valid send it to that address.
- else send $B$ to all other outgoing ports (**flooding**).

The problem with this approach is that it can creates loops in the network, because if we have two switches, S1 and S2, interconnected; S1 received something and flood to switch S2 and S2 also does not know the location, it will flood back to S1, this process is fast and grows exponentially so it can consume a lots of bandwidth and leads to catastrophic congestion. (**broadcast storm**, one of the goal of protocol is to avoid this).

#### Spanning Tree Protocol (STP)

In this protocol, it defines how can switches build a spanning tree (subgraph that connects all vertices with no loop) in an all connected LAN; with automatically setup and failure repair by exchanging **control-plane** messages. It builds its tree with following intermediate steps:

1. Find root.
2. Compute edges (min hops to root).
3. Support forwarding between any pair of LANs.

**Computing Logical Topology**:

1. Root ID is pre-assigned;

2. Nodes just agree that the min id will be the root and send message to each other.

   - broadcast at the beginning and when received a better message, change own configuration message and broadcast again.
     - Better message if:
       - smaller root id;
       - then shorter distance to root;
       - then lower sender id;
       - then lower port number. (broker)
   - configuration message: **(root_id, distance_to_root, sent_from_id)**.

3. Decide which port to block to form the spanning tree.

   - Each node send configuration message to root, keep only the **designated port** (port that has shortest path to LAN or cable to the root, i.e. the port received the best configuration message).

     - So now port will have a status:
       - **Root (R)** best external message received.
       - **Designated (D) (initial)** internal message is better than best message received in this port.
       - **Blocked (B)** internal message is worse than best message received on this port.

     <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/02/upgit_20220223_1645613196.png" alt="image-20220223104634734" style="zoom:50%;" />

   - Note the node does not stop and assign port status after it receive all configuration messages, instead it continuously update its port status based on configuration received and internal configuration, therefore configuration message is never blocked, so that we may update a port from blocked to designated.

Now we have our root decided, but topology (network) may change; so the root switch sends periodically (2 seconds recommended in 802.1d) with parameter *hello time*, and other switch send on all designated ports when receive roots' message.

All configuration messages are stored in a table with age field (time-to-live, there is a threshold max age, which is 20 seconds in 802.1d), discard when expired or newer received.

If the ethernet fails, the node would stop hearing message from one of the port, so set that port back to initial state (designated), if that port is root port, we update to a new root port and update own message.

If the ethernet backup again, LANs may ends up with loops, therefore STP adds a new port status called **pre-forwarding**, when the port is either:

- changing from blocked to designated.
- newly connected.
- on freshly-powered switch.

When it is in that state, it send configuration message and transition to blocked and root states as if it were designated, but does not forward data to avoid loops; until the port is assigned a new status after appropriate configuration message is received. Note that a port may ended up in pre-forwarding state forever.

- e.g. this can happen to root's port because it may not receive a better / worse configuration message because its internal message is already the best configuration message.
  - So we set a forwarding delay that is enough for the entire spanning tree to reform (in in practice: twice the maximum transit time across the extended LAN.), (30 seconds in 802.1d). After it timeout, we set port to be designated.

### Internetwork Protocol

Now what about the internet? We have STP that handles small group of nodes well, can we scale it for internet? Of-course not, if we use STP in the internet:

- All hosts hear all traffic during flooding, **does not scale well**.
- Even if we just track one host per LANs, that also means each switch needs to learn where every host is, and we will **need to keep lots of states**, and **packets still floods** until switches learns them.

We need to add another hierarchy. Lets connect hosts together to form **Local-Area Networks (LANs)** through ethernet, WIFI, etc... Then we add router to each LAN, connecting them to form a **Wide-Area Network (WAN)**, this is the extra layer that transfer packets between LANs, and we need a **internetwork protocol** for them.

Recall within the LAN we have flat MAC addresses, and new in WAN we will have structured IP address, which means address in the same LAN will have address close to each other, IPv4 is a 32-bits number, commonly represented as 4 number (dotted-quad notation, a.b.c.d). IP address had three attempts

1. Originally 8-bits network identifier, 24-bits for hosts.

   - assumed 256 networks, simply not enough
2. Next we try classful addressing, class A, B, C

   1. 0, 7-bits for network, 24-bits for hosts (large block taken by IBM, MIT, HP...)
   2. 10, 14-bits for network 16-bits for hosts (medium size organizations, huge demands)
   3. 110, 21-bits for network, 8-bits for hosts (small organizations, too small)
3. Today we have **Classless Interdomain Routing (CIDR)**

   - flexible boundary between network and host address (separate by IP mask), result in high address assignment efficiency. Written as network/mask length: 12.4.0.0/15.
   - They are allocated continuously and hierarchically, forwarding based on prefixes.
   - (I think now we use Ipv6)

### Tying the Link & Network Layer

- What IP address should the host use?
- How to contact the local DNS server?
- How to send packets to destinations?
  - Local and remote



- **Dynamic Host Configuration Protocol (DHCP)**
  - The end hosts learns **IP address**, **IP mask**, **local DNS server** and **gateway to internet** locally.
- **Address Resolution Protocol (ARP)**
  - Enable hosts reach local destination by providing mapping between IP and MAC address.

The key ideas:

- broadcast when in doubt: send query to all hosts in the LAN.
- cache information for some time: reduce overhead and allow communication
- soft state: eventually forgets, TTL, refresh/discard, provide robustness to unpredictable change.
  - Client may not release their IP address due to crashes or buggy software.
    - Short lease time: return inactive addresses quickly; long lease time: avoid overhead of frequent renewal (trade-off).

#### DHCP

1. At the beginning, client knows nothing about source and destination IP / MAC, so he broadcast a DHCP discovery message: source: 0.0.0.0; destination MAC: ff:ff:ff:ff:ff:ff and destination IP: 255.255.255.255;
2. Then either a **DHCP server** or **relay agent** can reply with a DHCP offer message:
   - Proposed IP address, netmask, gateway, DNS server and lease time
   - A relay agent forward configuration message to a remote DHCP server and their replies to client.
3. Client then accepts one of the offers by sending DHCP request.
4. Server confirms with DHCP ACK.

All four messages are broad-casted:

- Discover broadcast: client does not know DHCP server's identity.
- Offer broadcast: client doesn't have an IP address yet.
- Request broadcast: so other servers can see.
- ACK broadcast: client still doesn't have an IP address.

#### ARP

Now the host has information about its own, and it can send packet directly if it is local, but it does not know destination MAC address yet. To build such mapping (table containing IP and MAC pairs) it uses **Address Resolution Protocol (ARP)**, which basically is:

1. Broadcast who has this IP address X.
2. That guy with IP X response with IP address X is at link layer address Y.
3. Host cache this information in the ARP table.

Now what if after getting the address from DNS server and host find that the destination is not local (checked using netmask), then it will send the packet to a router (default gateway for the LAN, known from DHCP).

Note that this is not secure because any node can say whatever they want and send a ARP reply, which can result in **impersonation** and **man-in-the-middle attack**. Also note that the attacker does not need to win the race with real server, it can just poison host's cache due to host's optimization.

#### Sending Packet From Host A to B

1. Finds IP from local DNS server
2. Check if host is local using netmask
   1. yes: just send it directly to B
   2. no: 
      1. encapsulate IP packet in link layer head and send it to router
      2. router consult forwarding table and send it.

## Internet Protocol (IP)

Goal:

- connect heterogeneous networks together
  - unifies architectures of network of networks
  - run over ip = run over any network
  - support ip = run any application

### Header

Things to consider:

- max size of the packet?
  - many different max frame size in link technologies and we do not want to design for the smallest frame size because we do not know what is the smallest one in our route. So we use **IP fragmentation** that splits the packet.
    - but after we send the packet fragment when we encounter a link with smaller maximum transfer unit (MTU), we need to put it back together; since packets may take a different route, only the end host is guaranteed to receive all packets. 
- what should the receiver do?
  - To reply, we need **source ip address**, packet can provide a common way to retrieve it, but include it in every packet can be redundancy, it can just be send once in the start of the conversion between two hosts. (but we add it anyway).
  - Many protocol on top of IP, need to tell receiver what protocol we are using so that they can parse it, so we need to include a **next protocol id** in the header: TCP (6), UDP (17), tunneling: GRE and control: ICMP.
- what happen if packet gets corrupted?
  - some errors escapes CRC detection, we can add more error detection but only the application knows how much overhead is appropriate; but a checksum is still useful for protecting misrouted packets (or corrupted TTL), so we add a simple checksum for ip header only; we should not over-engineer low layers so it does not cover payload.
- what if router is confused?
  - router can be confused when there is a loop in the route:
    <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/03/upgit_20220312_1647099068.png" alt="image-20220312153107266" style="zoom: 50%;" />
  - The solution is to add a counter (time-to-live TTL) in the packet header, and decrease by one at each counter, drop when reaches zero; and send time-exceeded control message (ICMP) back to source.
- order/importance of different packets?
  - packets will need to queue when too many packets arrive at a router, it is easy to add a priority in the header but it is hard to prioritize them, this is a business/money thing, and hard to decide who to pay when packets traverse through multiple network.

At the end, we have:

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/03/upgit_20220312_1647100301.png" alt="image-20220312155141907" style="zoom:67%;" />

### Forwarding at the Network Layer

Routers are more powerful than switches in the sense that they can use IP addresses to foward packets across the internet. It:

- consists of a set of network interfaces where packets will arrive and depart;
- communicates with other routers to compute WAN path and forward packets to corresponding output interface.
- store information on the destination host, does not store per-host information --- scales well.
  - relies on structure of IP address.
- forward packets independently --- failover if paths fail 故障转移;

Providers tell internet what addresses are accepted, and when a packet with a particular address is received, the provider with most specific address is chosen to send the received packets. This means that if you want to do e.g. load balancing using multiple providers, all your providers must tells the same specificity of ip address range, otherwise all of them will be send to the provider with most specific address.

- host: endpoints running applications, at least one interface
- router: intermediate system with multiple interface
- subnet: point-to-point (cable between two routers) or shared LAN where hosts and routers are connected with their interface.

#### Computing the Table

Router build routing table with three entries: 

- **destination** IP prefix only, subnet's address.
- **outgoing interface** where to forward the packet.
- **metric** cost to reach destination, depends on interfaces' metrics (set by net admins)

When received a packet, router will find the longest prefix match, if no entry found, the packet is dropped $\rightarrow$ so how do we populate the packet?

Keeps a routing table; Applies LPM (longest prefix match) to incoming packets

- Initialize table for directly connected subnet; cost are minimal as they are connected.
- Routing protocols add information on remote destinations

In the routing protocol, it needs to define:

- Information and message exchanged;
- and method to determine the next-hop

In addition, we have the following requirements for internet:

- Good performance, possibly both time and space
- No central knowledge needed (e.g. graph algorithm such as Dijkstra cannot be used)
  - A distributed algorithm is needed for this shortest path problem
- Changing network
  - need fast reconvergence
  - avoid loops
    - it amplify traffic and leads to congested links
    - cannot rely on ttl to expire --- too late
    - possible temporary disconnection

- Automated path computation
- dealing with failures: coherent with topology, new route when old is disrupted
  - Need fast reconvergence to corrected path
  - Potential loops that amplify traffic and need to wait for TTL expire, but typically too late.
  - Possibly temporarily disconnections.

## Distance Vector Routing Algorithm

Compute shortest path distributed-ly that deals with topological changes.

- Distributed Bellman-Ford (DBF)

Tell everything you know to your neighbour periodically

When receive an announcement from neighbor that says they got a route for destination $D$, metric $m$ on interface $i$:

1. m += configured metric for interface $i$
2. route = lookup($D$)
3. if route not found
   - create new entry in routing table with destination $D$, metric $m$ and interface $i$
4. else if m < route.m  // we found better route
   - route.m = m; route.i = i

- neighbors incorporate announcement and propagate them further
- eventually converge and all routers will know next-hope for any destination

### Dealing with Failure

But what if we have a failure?

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220513_1652479102.png" alt="image-20220513225820143" style="zoom:50%;" />

Let say $D$ use $A$ to route to $B$, but $A$ finds that $B$ got a failure; but when $A$ send announcement to $D$, $D$ will ignores it as it thinks it is not a better route according to the algorithm on top, but there is an easy fix: update to newer metric if it comes from the same interface:

- step 3.5: if route.i == i then route.m = m 

### Dealing with Loop

OK, but what if $B$ found connection to $C$  is broken, but before it can send an announcement, $B$ received an announcement from $A$, saying he got a route to destination $C$? Now $B$ thinks $A$ got a route to $C$, but $A$'s route is actually using $BC$ link which $B$ knows is in failure! Even though $E$ got a route to $C$, after receiving broadcast from $B$ he will thinks that $B$ got a route with low cost, so he will not use that route.

Now, $A$ and $B$ will advertise to each other because they have inconsistent route, leading to long and painful convergence, and  transient forwarding loops, until their metric is higher than the one $E$ has in hand; and if there is no alternative path, it will loop to infinity.

The problem is that $A$ is announcing to $B$ that he got a route to $C$, even though he is using $B$ to route to $C$, he should never do that! We got two approaches:

- Split Horizon: $A$ does not announce to $B$ that he got a route if he is using $B$ for that route
- Poison Reverse: $A$ announce to $B$ that its distance to $D$ is infinity.

### Limitation

OK, but what if we have a scenario like:

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220513_1652480168.png" alt="image-20220513231608707" style="zoom: 67%;" />

Now $E$ found that $ED$ is broken, he sends announcement to $B$ but not yet reach $C$, now, $B$ updates its table, but then he received announcement from $C$ that $C$ got a path to $D$ ($C$ have not updated), then $B$ is happy and updates its table, and advertise its table, now we got infinity loop again!

> Fuck, what's next?

## Link State Routing

tell everybody what you know about your neighbourhood, flood local information network wide, by saying hello periodially $P$. Each hello packet contains sender id and list of neighbours which sender heard hello during period $D$, (e.g. $D=3P$). A adjacencies table of any two routers is built, up if they are hello-ed to each other, else down

- a Link State Advertisement (LSA) is sent to all neighbours when local change detected
- all LS router runs Flooding Protocol to bounce news.
- each LSA contains 
  - ID of the advertising router
  - sequence number
  - local link information: id and metric
  - message parameters: LS age, type, etc.
- received LSAs are stored in Link-State database, flood its information to others when updated

Its pseudo code are as follows:

1. if link not in database or LSA seq no. > stored seq no.
   1. store LSA in databse
   2. send LSA to all interfaces except the received interface
2. elif LSA seq no. < stored seq no.    // out of date, update neighbor instead
   1. send the stored LSA back to incoming interface
3. else:
   1. ignore LSA   // we already heard it

### Counting to infinity?

No, it does not count to infinity

### Tricky Case

Now if we have a network like this:

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220515_1652655214.png" alt="image-20220515235332895" style="zoom:50%;" />

Suddenly, $DE$ healed and $BC$ shutdown almost the same time, now, if $DE$ has been up in the past, then $D$ may still have the old cache that $BC$ is linked! He was not updated because $DE$ failed, and from $B$ and $C$ perspective they already updated their neighbour, so they will not update again.

but good news is that we can solve this by setting a timer to periodically update neighbour even though the database information did not change: OSPF sends new LSAs every 30 mins, even for unmodified links, but this is a trade-off: sent often means overhead and less often means too slow for update. The root case for this is flooding is not always efficient!

So the actual solution is: routers exchange LS database summaries when they form new adjacencies.

Now each router run a dijkstra algorithm locally to find the shortest path

- polynomial time
- $O((|V| + |E|) log2 |V|)$ or $O(|E| log2 |V|) $ when all vertices are reachable
  - Note most network are sparse so: $– |E| << |V|^2$ which means dijkstra algorithm is almost linear.

## Distance Vector VS Link State

Link State vs. Distance Vector

- LS generally scales and performs better than DV 
  - no routing bouncing and counting to infinity, hence, faster convergence after topology changes
- However, LS is more complex to implement than DV – LSAs’ sequence numbers crucial to protect against stale announcements
  - adjacencies have to be established and maintained
  - link state database + flooding are needed
- Flooding status of all links seems costly, but actually reasonable for tens and tens of routers
  - vanilla LS doesn’t scale indefinitely. e.g., for thousands of nodes – yet, scalability can be improved with hierarchy 
- In practice, LS is more popular than DV – LS is more commonly used, especially in large (ISP) networks, where routing is critical
  - Use of DV is more niche • “small” enterprise networks, wireless ad-hoc ones, ...

### Real World is Complex

LS does not solve all problems, e.g. transient loops during convergence. But its connectivity is of utmost importance: it is money! It attracted many industrial and research efforts. E.g. loop-free alternate (LFA) and its variants and progressive metric increment

Network has many more requirements e.g. 

- avoid congestion
- enforce firewall traversal
- additional protocols e.g. MPLS
- traffic engineering systems may want to have routing on non-shortest paths

Big players like google and microsoft starts to develop their own routing systems e.g. Google B4, Microsoft SWAN and Google BwE.

## Inter-domain Routing

Each domain is called an autonomous system (AS), each known by a unique 32 bits address, each owning a handful of IP prefixes.  They are assigned by Regional Internet Registries (RIR) and owned by IANA. There are about 70,000 domains as of Sep 2020 (see www.cidr-report.org)

- Each AS cooperate to find optimal paths is not possible
- Each ISP interconnected equally is also little correspondence in reality
- In reality: ISPs have tiers, worldwide then regional... Routing follows money! Support per-AS policies
  - Basic model: customer-provider, pay for connectivity. Or Peering, two ISPs forward own traffic to each other, no exchange of money
  - Peer offer better performance and attract customers
    - Tier-1 ISP must peer to build compute global routing tables!
  - But Peering does not let ISP charge everybody, and need to agree on asymmetric traffic loads
    - Nobody knows how traffic patterns will change after establish a new peering

So, DV and LS are Interior Gateway Protocols (IGPs) that only runs within a single AS!

## Border Gateway Protocol (BGP)

Goals:

- Scalability: unique IP, no consult to central authority
  - Routers track networks / IP prefixes not hosts, IP is assigned hierarchically most of the time
  - \>70k ASes and >800k announced prefixes
- Enforce policies not optimal performance!
  - ASes are competitors and routing must reflect commercial agreements
  - Need to cooperate but under competitive pressure
    - BGP designed to run on successor to NSFnet, the former single, government-run backbone

The problem with Inter-domain protocols is:

- insufficient scalability – DV and LS cannot scale to Internet routing
  - prohibitive cost for LS flooding
  - loops and slow convergence for DV
- No support for policies
  - cannot reflect commercial agreements using DV and LS which compute shortest paths 

So, BGP! An Exterior Gateway Protocol (EGP) that computes paths spanning multiple ASes

- It is the de-facto inter-domain routing protocol which allows to route between ASes, with policies
- It is additional and complementary to intra-domain routing

> ++

- Customer routes (remunerative) > peer routes (neutral) > provider routes (cost)

> ++

### Protocol

It is **stateful** and **connection-based**, between routers. Runs over **TCP** port 179, because it provides reliable delivery and no need for periodic re-announcement of the same routes.

It is to enable reachability and supprting policies

1. connects by sending a OPEN message
2. upon new connection, exchange all active routes in their tables
   - can takes minutes, depending on size and implementation
3. they send two main types of messages
   1. announcement: new/update route
      - contains attributes that describe characteristics of announced route: to support policies, based on them, routers can:
        1. avoid inter-domain loops
        2. apply custom route filtering and ranking
        3. influence other routers' decisions
      - human operators need to configure these routers so that they implement the intended policies, which is challenging and error-proneDD
   2. withdrawals: retraction of previous route
      - only include IP prefix

Internals of a BGP router

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220516_1652657968.png" alt="image-20220516003928199" style="zoom:50%;" />

### BGP decision process

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220516_1652657986.png" alt="image-20220516003946871" style="zoom:50%;" />

- **Local preference** is a numerical value with local meaning. All BGP routers compare LP across known routes for same destination, can be used to reflect commercial relationships

  - typically, LP for customer routes > LP for peer routes > LP for provider routes

- **AS-PATH**: BGP is a Path-Vector Protocol, 

  - announcement contains full list of AS numbers along the path to prevent inter-domain forwarding loops (discard route if own AS number (ASN) is in the AS-PATH)
  - it is the first tie-breaker after LP, among routes with min LP, prefer shortest AS-PATH

- **Origin**: Prefer IGP-Originated Routes

  - Usage: mostly historical

  - Origin indicates how BGP learned about the route

    - IGP: route interior to the originating AS
    - EGP: originated via the (obsolete) EGP protocol
    - Incomplete: unknown or learned some other way

    fixed preference: IGP > EGP > incomplete

- **Multi-Exit Discriminator (MED)**: Choose between Multiple Exit Points

  - Tier-1 and Tier-2 ISPs often span differen geographic regions, with same LP and AS-PATH
  - One AS may prefer a particular transit point to save money
  - It is used  is used to express transit point preferences

  One AS can use MED as advertisement to other AS

  - MED is an interger cost and router should choose the lowest MED advertised
  - other AS do not need to honor it, unless motivated by some financial settlement
  - often prefer shortest-exit routing: get packet onto someone else backbone as quickly as possible, and result in highly asymmetric route.

Actually BGP is two protocols

- eBGP: external BGP advertise routes between ASes
  - routes from AS neighbours
- iBGP: internal BGP propagates external routes inside a single AS
  - routes from iBGP routers in the same AS

Each router shares its best route internally to its own AS, select the best route, if best route is eBGP, then disseminates (传播) in iBGP.

#### iBGP

It selects the egress 出口 points. Its goal is to:

- ensure visibility: each router in AS must get at least one of the best eBGP routes: an eBGP route surviving all tie-breakers explained so far (?)
- loop-free forwarding

**Full-mesh**

- each router floods selected eBGP routes to all other iBGP routers
- Flooding done over TCP, using intra-AS paths provided by IGP, this is different from LS flooding!

Each router pushes all its best routes learned via eBGP to all iBGP routers 

Internal routers know all the best eBGP routes

- Pro: simple 
- Con: scales badly wrt number r of routers iBGP connections: O(r^2)

more scalable iBGP configurations are route reflection, and iBGP confederation

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220516_1652659153.png" alt="image-20220516005912909" style="zoom:50%;" />

> ++ Routers combine IGP and BGP tables

### Final Remarks

- Tier-1 ISPs have no connectivity provider. Hence, all tier-1 ISPs must peer with one another 
- The Internet tier-1 is a full mesh of (eBGP) connections
  - True tier-1 ISPs do not pay for peering, and do not buy transit from anyone
  - A few other large ISPs peer with all tier-1 ISPs but pay settlements to one or more of them

- For Internet to be connected, all ISPs who do not buy transit service must be connected in full mesh!

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220516_1652659549.png" alt="image-20220516010549843" style="zoom:50%;" />

ISP with no transit provider as of 2015

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220516_1652659399.png" alt="image-20220516010319736" style="zoom:50%;" />

- 10/2005: Level 3 de-peered Cogent
- 3/2008: Telia de-peered Cogent
- 10/2008: Sprint de-peered Cogent
  - –lasted from 30th October – 2nd November, 2008
  - 3.3% of IP prefixes in global Internet behind one ISP partitioned from other, including: NASA, Maryland Dept. of Trans., New York Court System, 128 educational institutions, Pfizer, Merck, Northup Grumman, ...



- Inter-domain routing chiefly concerned with policy
  - Economic motivation: cost of carrying traffic
  - Different relationships demand different routing: customer-provider vs. peering 
- BGP: stateful, path-vector routing protocol
  - Scalable in number of ASes
  - Route attributes support policy routing
  - Loop-free at AS level
  - Shortest AS-PATHs preferred, after policy enforced

Reality is more complex!

- Inter-domain policies can be more than customer-provider and peering, e.g. regional transit, or extra tier-I resilience
- BGP is powerful enough to support more policies than the ones we have discussed, can bypass some decision process
- hard to configure
- arbitrary policies?
  - slower path exploration
    - Some destination dies, all ASes lose direct path and switch to londer paths, eventually withdrawn
  - Outages
    - <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220516_1652659876.png" alt="image-20220516011116412" style="zoom: 80%;" />

> dyn.com/blog/widespread-impact-caused-by-level-3-bgp-route-leak/

BGP itself is far from perfect

- Long convergence after failures
  - distributed computation of best policies-compatible routes -> many interactions between many routers 
- No native security
  - e.g., BGP hijacking: malicious AS can pretend to own prefixes of other ASes, to blackhole/intercept traffic
- Overlooks performance metrics and QoS
  - real problem for content providers: e.g., Google espresso [2017], FB edge fabric [2017], ...

## Miscellanies

- traceroute: give destination, show hops, router ip
- whois: show organization given ip
- nslookup: show ip given name
- ipconfig: show network interface
