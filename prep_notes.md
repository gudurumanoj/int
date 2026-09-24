## Prep Notes

### Resume

Contributed to building a multilingual tokenizer for 22 Indic languages, improving average fertility score by 39.5% over
LLaMA4 and 18% over Sutra, yielding 44% inference throughput gains
- Gotta see what to do and tell here in terms what's roughly done, what/where I've contributed


---

Built a memory efficient DeepSeek Math inspired pipeline for high-quality math data extraction from the web
- Should describe deepseek math pipeline, what engineering difficulties I faced, how I solved them and what all I've learnt here to make it memory efficient
- COme up with a plan, had to right some custom multithreaded heap datastructure class implementation using python multiprocessing library
- Then add memory kept blowing up becuase of some lazy importing and copy on read something -- dont remember this exactly, should decide what to tell and frame here


---

Developed a novel loss function for multilabel classification prioritizing tail-label performance on skewed distributions
- Should once describe what metalearning is, coding a basic metalearning based loop in pytorch and understand how the gradients flow in this setup
- Gotta tell about focal loss, asymmetric focal loss and weighted ce/focal loss for imbalanced distributions

---

Developed ML models for short-term price movement forecasting, evaluating across classification and calibration metrics
Reduced inference latency by ∼100x via systematic model optimization while keeping accuracy within 5% of baseline
- Internship what exactly/roughly done
- How optimised dataloading and training in a modular way to be able to queue experiments and all
- How model trimming and latency reduction happened
- What all data/result analysis I did and all

---

Implemented priority-based scheduling and lazy page allocation based memory management with page fault handling
Added kernel-level multithreading with synchronization primitives for safe concurrent execution of processes
- Since Im an os ta, I should be able to answer every faqs from os and specifically how these work
    - Process v thread (what all properties differ)
    - Scheduling
    - Virtual v physical address space
    - What happens at a context switch
    - Lazy page allocation
    - Page fault handling
    - Paging (page table, segmentation)
    - Multithreading (what happens on a fork etc)
    - Synchronization primitives
    - Locks and condition variables(spinlocks, mutexes, semaphores)
    - Filesystem (inode, file descriptor, open, read, write, close)
    - IPC (pipes, shared memory, semaphores, message queues, sockets)

---

Designed a distributed application-layer protocol over TCP sockets for serverless peer-to-peer file transfer in C++
Implemented chunk-based transfer with MD5 integrity verification using raw socket programming (arpa/netinet libraries)
- How handled deadlocks and race conditions here
- How to handle large file transfers efficiently
- How I designed peer to peer and messaging 
- Networks brush up
    - TCP v UDP
    - TCP
    - UDP
    - Layers of network stack (application, transport, network, data link, physical)
    - How vpn works
    - How sockets work
    - How routing works

---

Multi-Language Question Answering System
Built a multilingual extractive QA system supporting Telugu, English, and Hindi via cross-lingual transfer learning
Analyzed impact of training data size and language similarity on extraction accuracy across the three languages
- BERT
    - How qa is done via using bert
    - How its traiend (mlm method)
    - Bert architecture
    - Shapes at every step
    - What experimetns actually did to analyse traing data size and language similarity across the three languages?

---

(De)Noise — Speech Enhancement GAN
Implemented the SEGAN paper for speech noise reduction via adversarial training with convolutional residual blocks
Achieved measurable SNR improvement on noisy speech benchmarks by tuning residual block depth and training schedule
- Should be able to answer every faqs from speech processing and specifically how these work
    - SEGAN
    - Why residual blocks
    - What adversarial trainig done?
    - Whats the encoder decoder architecture?
    - SNR improvement by how much? 
    - How tuned residual block depth and training schedule? What were the changes from the paper or jsut straight up its the paper??