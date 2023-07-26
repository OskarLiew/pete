---
alias: Concurrency, Asynchronous code
---

**Concurrent computing** is when multiple computations are executed during overlapping time periods, instead of sequentially. It is related, but distinct from [[Parallel Computing]], where execution occurs at the exact same time on multiple processors to speed up computations. Concurrent programming instead consists of process *lifetimes* overlapping, but execution must not happen at exactly the same time.

## Types of concurrent computing

![[Concurrency.png]]

### Parallel Computing

In [[Parallel Computing]], the goal is to speed up a *CPU-bound task* by splitting the task on multiple cores through multiprocesssing.

### Threading

[[Threading]] is a concurrent execution model where multiple threads take turns executing tasks. One process can consist of multiple threads. Threading is suited for *IO-bound* tasks, where computations can be performed while waiting for input/output to complete.

### Async IO

[[Async IO]] is a single-threaded, single process design that uses cooperative multitasking to give a feeling of concurrency. Coroutines (a central feature of async IO) can be scheduled concurrently, but they are not inherently concurrent.

---

# References

1. <https://en.wikipedia.org/wiki/Concurrent_computing>
2. <https://realpython.com/async-io-python/>
