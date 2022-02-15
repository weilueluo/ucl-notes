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

For the receiver, he need to send one block ACK that tells the sender what are the packets has been received.

- receiver needs to be able to buffer 64 packets.
  - Note if one packet is dropped, the sender can only resend that packet in the next time because there is no space in the buffer to put more packets.
- block ACK is 64-bits vector where $i$-th bit indicates whether that packet is received.
  - Now this block ACK is critical because if it is corrupted/missing, sender needs to resend all the packets after timeout.
    - We can request block ACK again after timeout.
- Stop-and-wait is still slow for long RTT, because even if we increase the size for one go, we still need to wait for the block ACK from receiver
  - Now let's allow the receiver to send ACK asynchronously, instead of wait for everything. (? is this true?)
  - 

