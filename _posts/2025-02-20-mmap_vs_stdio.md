---
title:  "Memory-Mapped I/O vs Standard File I/O"
mathjax: true
layout: post
categories: media
---

When dealing with large files, efficient file handling can make a significant difference in performance. There are two common approaches: memory-mapped I/O (mmap) and standard file I/O (std). In this blog post, I'll explore a Rust program that benchmarks both methods, analyzing their strengths and weaknesses.

---


## The Rust Program

The provided Rust program benchmarks two approaches for truncating a file after the first newline character and counting lines:

- **Memory-Mapped I/O (mmap)**: Uses `memmap2::MmapMut` to map the file into memory and modify it directly.
- **Standard File I/O (std)**: Reads the file into memory, processes it, and writes the truncated content back.

It measures execution time for various operations and compares overall performance for different file sizes.

Code can be accessed [here](https://github.com/bernardo-sb/mmap_vs_sdio).

## MMAP vs STDIO

Before getting into the benchmarks, let's understand the underlying concepts behind both approaches.
Memory-Mapped I/O (MMAP) and Standard File I/O (STDIO) work in fundamentally different ways.

mmap is a technique that maps a file directly into the process's virtual memory space, thus the file becomes an extension of the program's memory space.
Access patterns are largely different between the two approaches. Standard I/O explicitly reads/writes data using system calls. In contrast mmap accesses the file's contents directly as if it were memory - dereference pointers and use normal memory operations, which allows sharing data efficiently between processes.
Standard I/O requires data to be copied twice - first from disk to kernel buffer, then from kernel buffer to the program's buffer. With mmap, data is loaded directly into a process's address space when needed, with page faults handling the actual disk I/O. 
With mmap, there's no need to make explicit system calls for I/O. When mapped memory is modified, the operating system automatically handles synchronizing those changes back to the disk. 


**Memory-Mapped I/O (MMAP)**

- In Rust we can use the `memmap2` crate to map a file directly into the process's virtual memory.
- The OS loads only the necessary parts (pages) of the file into RAM on demand, reducing memory overhead.
- Modifications are applied in-place without additional read/write operations.
- Efficient for large files as it avoids loading everything into RAM at once.

**Standard File I/O (STDIO)**

- Opens the file and reads its contents into a buffer in memory.
- Performs operations (like finding newlines) on the in-memory buffer.
- Writes the modified data back to the file, which can be slow for large files.
- Prone to high memory usage if the file is large, as it loads the entire content into RAM before processing.


### Explanation of the Rust Program

[This Rust program](https://github.com/bernardo-sb/mmap_vs_sdio) compares two different methods for truncating a file for two functions, one after the first newline (`\n`) character is found, and another after all lines in the file. The two methods used are:

1. **Memory-Mapped I/O (`mmap`)** - Uses the `memmap2` crate to map the file into memory for direct access.
2. **Standard I/O (`std::io`)** - Uses traditional file reading and writing.

Each method is timed for performance measurement using a higher-order function `measure`. The program prints the performance comparison at the end.

---

## **Breakdown of Key Components**

The program is composed of the following components:

- `Timing`: A struct for timing the execution of a function
- `measure`: A higher-order function for timing operations
- `truncate_with_mmap`: Function to truncate a file with MMAP and run an operation
- `truncate_with_standard_io`: Function to truncate a file with STDIO and run an operation
- `main`: The Main function for running the benchmark

### The `Timing` struct

struct Timing {
    operation: String,
    duration: std::time::Duration,
}
```

The `Timing` struct is composed of `operation` name and its `duration`.

### `measure`: A Higher-Order Function for Timing Operations
```rust
fn measure<F, T>(operation: &str, f: F) -> (T, Timing)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    (result, Timing::new(operation, start.elapsed()))
}
```
`measure` is a **higher-order function**, it takes another function `f` as an argument. It records the start time, executes the function `f()`, and then returns the result along with the elapsed time. `FnOnce` ensures that `f` is only called once.

A more detailed explanation of `FnOnce` (from [Reddit](https://www.reddit.com/r/rust/comments/1fj38s7/whats_the_difference_between_fnonce_fn_and_fnmut/)):

>Closures have a capture environment and `FnOnce` takes it by value therefore after the closure runs once, the environment is gone. `Fn` takes the environment by immutable reference so it can be called an arbitrary number of times but can't mutate the captures. `FnMut` takes the environment by mutable reference and can be called many times and can mutate the captures

---

### Opening a File Using `OpenOptions`

Both `truncate_with_mmap` and `truncate_with_standard_io` open a file using `OpenOptions`:

```rust
OpenOptions::new()
    .read(true)
    .write(true)
    .open(filename)
```

This allows both **reading** and **writing** to the file.

---

### Memory-Mapped File Handling (`truncate_with_mmap`)

#### Opening and Mapping the File

```rust
let (mmap, timing) = measure("create mmap", || unsafe { MmapMut::map_mut(&file) });
let mmap = mmap?;
timings.push(timing);
```
**`MmapMut::map_mut(&file)`** maps the file into memory. The **`unsafe` block** is required because `memmap2` works with raw memory pointers.

#### Finding the First Newline Character

```rust
let (ix, timing) = measure("find newline (don't fill zeros)", || {
    let ix = mmap.iter().position(|&x| x == b'\n').unwrap_or(mmap.len());
    ix
});
timings.push(timing);
```

Iterates over the mapped memory and finds the first occurrence of `\n`. For this case, we will truncate the file up to and including the found `\n`.

#### Truncating the File

```rust
let (_, timing) = measure("truncate file", || file.set_len(ix as u64));
timings.push(timing);
```

 `file.set_len(ix as u64)` truncates the file up to the found newline.

---

### Standard I/O File Handling (`truncate_with_standard_io`)
#### Reading the File into a Buffer

```rust
let (contents, timing) = measure("read file", || {
    let mut contents = Vec::new();
    file.read_to_end(&mut contents).map(|_| contents)
});
let contents = contents?;
timings.push(timing);
```

Reads the entire file into a `Vec<u8>` buffer.

#### Finding Newline and Creating Truncated Buffer

```rust
let (truncated_contents, timing) = measure("find newline and create new contents", || {
    let ix = contents.iter().position(|&x| x == b'\n').unwrap_or(contents.len());
    contents[..=ix].to_vec()
});
timings.push(timing);
```

Finds the index of the first `\n`, using `position()`, and creates a new buffer with data up to and including it.

#### Writing the Truncated Buffer to the File

```rust
let (_, timing) = measure("write new contents", || file.write_all(&truncated_contents));
timings.push(timing);
```

Writes the truncated buffer back to the file.

---

### Counting Newlines (Commented-Out Alternative)
Both I/O alternatives have a commented-out alternative that counts newlines instead of truncating:
```rust
let (count, timing) = measure("count lines", || {mmap.iter().filter(|&x| *x == b'\n').count()});
```

Uses `iter().filter()` to count all occurrences of `\n` in the file.

---

### Replaceing `,` with `;` (Commented-Out Alternative)

Both I/O alternatives have a commented-out alternative that replaces `,` with `;` in the file:

```rust
let (_, timing) = measure("replace , with ;", || {
    mmap.iter_mut().for_each(|x| {
        if *x == b',' {
            *x = b';'
        }
    });
});
```

`iter_mut()` iterates over a mutable slice and applies a closure to each element. Inside the closure substitution is done inplace, `*x = b';'`.

### Flushing the File

```rust
let (_, timing) = measure("flush file", || file.flush());
timings.push(timing);
```

Flushing the file ensures that all changes are written to the disk. This only applies to mmap.

---

### Performance Comparison and Output

The program prints timing results:
```rust
println!("Memory Mapping: {:?}", mmap_total);
println!("Standard I/O: {:?}", std_total);
```
- It compares the total duration of each approach and prints the difference.

---

## **Summary**
| Feature            | `truncate_with_mmap` (Memory Mapped I/O) | `truncate_with_standard_io` (Standard I/O) |
|--------------------|--------------------------------|----------------------------------|
| **File Opening**   | `OpenOptions::new().read(true).write(true)` | `File::open(filename)` |
| **File Reading**   | Direct memory access via `mmap` | `file.read_to_end()` |
| **Finding Newline** | `mmap.iter().position(|&x| x == b'\n')` | `contents.iter().position(|&x| x == b'\n')` |
| **Replacing `,` with `;`** | `mmap.iter_mut().for_each(|x| { if *x == b',' { *x = b';' } });` | `contents.iter_mut().for_each(|x| { if *x == b',' { *x = b';' } });` |
| **Truncation**     | `file.set_len(ix as u64)` | `file.write_all(&truncated_contents)` |
| **Performance**    | Faster for large files | Slower due to copying |

The **memory-mapped approach** (`truncate_with_mmap`) is generally faster because it avoids copying the file into a buffer, while **standard I/O** (`truncate_with_standard_io`) is simpler but may be slower for large files.


## Benchmark Setup

The benchmarks were performed on CSV files of different sizes:

1. **Large file**: 100 million rows, 100 columns (\~17.95GB)
2. **Small file**: 1000 rows, 10 columns (\~197.7KB)

### Results

Below are the results comparing the performance of memory mapping (mmap) and standard I/O operations when handling large and small files across three tasks:

1. Keeping the first line and truncating the file
2. Counting lines
3. Replacing content

The tests were conducted on datasets of varying sizes:
- **Large dataset**: 100,000,000 rows, 100 columns (~17.95GB)
- **Small dataset**: 1000 rows, 10 columns (~197.7KB)


### Task 1: Keep First Line and Truncate File

#### Large Dataset
| Operation                        | Memory Mapping | Standard I/O |
|----------------------------------|------------------------|----------------------|
| Open file                        | 39.375µs               | 38.5µs               |
| Create mmap                      | 18.292µs               | -                    |
| Find newline                     | 788.333µs              | 2.944084ms           |
| Flush mmap                        | 2.375µs                | -                    |
| Truncate file                     | 7.130666ms            | -                    |
| Read file                         | -                      | 48.562204666s        |
| Open file for writing             | -                      | 420.421ms            |
| Write new contents                | -                      | 34.125µs             |
| **Total Time**                    | **7.979041ms**         | **48.985642375s**    |
| **Difference**                     | **48.977663334s**      |                      |

#### Small Dataset
| Operation                        | Memory Mapping    | Standard I/O    |
|----------------------------------|------------------------|----------------------|
| Open file                        | 26.625µs               | 16.042µs             |
| Create mmap                      | 26.5µs                 | -                    |
| Find newline                     | 3.166µs                | 1.208µs              |
| Flush mmap                        | 3.75µs                 | -                    |
| Truncate file                     | 35.167µs               | -                    |
| Read file                         | -                      | 42.5µs               |
| Open file for writing             | -                      | 44.875µs             |
| Write new contents                | -                      | 14.792µs             |
| **Total Time**                    | **95.208µs**           | **119.417µs**        |
| **Difference**                     | **24.209µs**           |                      |

### Task 2: Count Lines

#### Large Dataset
| Operation                        | Memory Mapping         | Standard I/O         |
|----------------------------------|------------------------|----------------------|
| Open file                        | 25.292µs               | -                    |
| Create mmap                      | 18.125µs               | -                    |
| Count lines                       | 304.405998334s        | -                    |
| Flush mmap                        | 39.919875ms           | -                    |
| Truncate file                     | 69.167µs               | -                    |
| **Total Time**                    | **Process killed**     | -                    |

Unable to count lines in large files using standard I/O due to memory constraints.
macOS kills process when virtual memory is exceeded; mmap does not load the entire file into RAM at once. Instead, it maps the file into virtual memory and only loads pages as needed.

#### Small Dataset
| Operation                        | Memory Mapping    | Standard I/O    |
|----------------------------------|------------------------|----------------------|
| Open file                        | 13.875µs               | 15.208µs             |
| Create mmap                      | 27.625µs               | -                    |
| Count lines                       | 1.406ms               | 1.334292ms           |
| Flush mmap                        | 2.666µs                | -                    |
| Truncate file                     | 11.459µs               | -                    |
| Read file                         | -                      | 41.666µs             |
| Open file for writing             | -                      | 35.25µs              |
| Write new contents                | -                      | 30.625µs             |
| **Total Time**                    | **1.461625ms**         | **1.457041ms**       |
| **Difference**                     | **4.584µs**            |                      |

### Task 3: Replace Content

#### Large Dataset
| Operation                        | Memory Mapping  | Standard I/O  |
|----------------------------------|------------------------|----------------------|
| Open file                        | 56.833µs               | 45.416µs             |
| Create mmap                      | 34.75µs                | -                    |
| Replace                           | 150.786251792s        | 180.585382625s       |
| Flush mmap                        | 641.97ms              | -                    |
| Truncate file                     | 68.25µs                | -                    |
| Read file                         | -                      | 8.919290083s         |
| Open file for writing             | -                      | 17.342167ms          |
| Write new contents                | -                      | 117.356627166s       |
| **Total Time**                    | **151.428381625s**     | **306.878687457s**   |
| **Difference**                     | **155.450305832s**     |                      |

#### Small Dataset
| Operation                        | Memory Mapping    | Standard I/O    |
|----------------------------------|------------------------|----------------------|
| Open file                        | 31.792µs               | 39.459µs             |
| Create mmap                      | 23.75µs                | -                    |
| Replace                           | 1.663917ms            | 1.039125ms           |
| Flush mmap                        | 489.458µs              | -                    |
| Truncate file                     | 18.333µs               | -                    |
| Read file                         | -                      | 111.916µs            |
| Open file for writing             | -                      | 62.417µs             |
| Write new contents                | -                      | 29.042µs             |
| **Total Time**                    | **2.22725ms**          | **1.281959ms**       |
| **Difference**                     | **945.291µs**          |                      |

### Analysis
Memory mapping significantly outperforms standard I/O for large files, particularly in reading and writing operations. However, for small files, the difference is minimal, with standard I/O occasionally being faster.

When counting lines in large files, memory mapping faced limitations due to virtual memory constraints, leading to process termination.

In replace operations, memory mapping demonstrated superior performance for large files but was slightly slower for smaller ones.

For large files, memory mapping is highly efficient, especially when modifying content. However, for small files, standard I/O performs similarly or slightly better, making it the preferred choice for such cases.


## Why Memory-Mapped I/O Wins

1. **No Need to Load Entire File into RAM**: Only required pages are loaded, preventing crashes on large files.
2. **Virtual Memory Efficiency**: Pages are swapped in and out dynamically by the OS.
3. **Zero-Copy Optimization**: Avoids redundant data copying between kernel and user space.
4. **Fast Seeking and Navigation**: Allows instant access to file contents without reloading data.

## Bottom Line

Memory-mapped I/O outperforms standard I/O significantly when handling large files, making it a powerful technique for file processing.
When dealing with massive files and when efficient reads/writes are necessary, memory mapping is a great option.

---

## Resources

- [GitHub](https://github.com/bernardo-sb/mmap_vs_sdio)
- [Memory-mapped file](https://en.wikipedia.org/wiki/Memory-mapped_file)
- [Memory-mapped I/O](https://en.wikipedia.org/wiki/Memory-mapped_I/O)
- [Virtual memory](https://en.wikipedia.org/wiki/Virtual_memory)
- [Rust Memory-Mapped I/O](https://docs.rs/memmap2/latest/memmap2/struct.MmapMut.html)
- [Difference between FnOnce, FnMut, and Fn](https://www.reddit.com/r/rust/comments/1fj38s7/whats_the_difference_between_fnonce_fn_and_fnmut/)
