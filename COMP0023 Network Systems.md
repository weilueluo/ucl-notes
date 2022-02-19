# Network Systems COMP0023

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

which does not hold on real data network, packet arrivals are bustier (instead of random) and the queue in router is finite, they have some upper bound for delay. But how much memory does  each router needs? (dimension switches' queues). Memory is relative cheap so can we use the worst-case memory size? No, because worst-case is orders of magnitude larger than average, and even if we can queue everything, the user will rather retry or just give up than wait for long time.

So we use average sized memory, but what to do when it is full (congestion)? We cannot ask the sender to slow down (send a quench message) because that will generate more traffic and the source may not be sending anything more. So when the queue is full, we just simply drop the packet, and sender will need to resend the packet (if he cares). This implicitly tells the sender that the traffic is congested now (possibility of slowing sender automatically to response to it), and this is what the internet does.

### Best Effort Delivery

Networks that never discard packets are called **guaranteed-delivery**, which potential for higher delay, and network such as the internet is called **best-effort**. Note that internet applications are required to build guaranteed delivery on-top-of best-effort delivery (e.g. email, trying again and again).

Implications:

- Packet loss.
- Duplication: duplicate send because sender did not receive a respond.
- Unbounded Delay: receiver sent a response but was delayed, so sender send again and received two responses as well, also causes duplication.
- Noise on links: packet can be corrupted, but error correction is mostly efficient and these packets are dropped.
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
- If all three assumptions holds, we just send messages one-by-one, end.

- Now if $1$ does not hold:

  - We may try to send ACK/NACK message to the sender to acknowledge them if the message has been corrupted, but this ACK/NACK message can also be corrupted, so it does not work.

  - Another way is to always resend when in doubt (doubt=NACK received), but now we may receive duplicates if ACK is corrupted.

    - We can introduce a 1-bit sequence number to resolve this, if we receive the same sequence bit message, that means we have duplicates so we just discard it. We can also drop NACK message here and use this sequence bit to tell the sender if we have receive the last packet or not (e.g. if we expect sequence 0 but did not receive it, then we send 1 back to the receiver, otherwise 0).

- Now if $2$ does not hold:

  - add a timer, resend when doubt, (doubt = timeout + NACK received).

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
- If packet 2 is lost and we received ACK 1 before sending packet 4 and we received another ACK 1, then we know that packet 4 has been successfully transmitted (?). (? what if out-of-order but none lost?)

The sender will need to resend all packets after the last packet that he was acknowledge (timeout / ACK from receiver), hence go back n.

- a timeout timer is started when a packet is sent, and cancelled when its ACK is received.
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
  - We **use cumulative sequence number**, increase monotonically, receiver can drop packets that has been received in order. Although this reduce the problem, we still need to set the sequence space for the number, we come back to this later

#### Preserve data ordering

- transport layer **segment** the data and **reassemble** at the receiver side, **mark each packet by its range of bytes in original data**, pass to receiver when all bytes before some point has been received.

#### End-to-end integrity

- Use **internet checksum over**:
  - **payload** protect against link layer reliability.
  - **transport protocol header** protect header sequence number and payload mismatch.
  - **layer-3 source and destination** protect against delivery to wrong destination.
- cannot protect against software bugs / router memory corruption (?), etc..
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
