---
title:  "The Actor Model in Rust"
mathjax: true
layout: post
categories: media
---

In this blog we tinker around the actor model in rust. It's a very interesting exercise given Rust's unique features.
Rust's strengths in memory safety and concurrency make it a great choice for building robust, concurrent systems. In this post, we’ll explore a program that implements an **actor model** in Rust using the asynchronous runtime **Tokio**. This example illustrates message-passing, state management, and graceful shutdown in a concurrent environment.

---


# Implementing Actors with Tokio: A Practical Guide

When designing systems for high concurrency and scalability, the actor model is a popular choice. It encapsulates state and behavior, communicating solely through message passing. Rust's async ecosystem, powered by **Tokio**, is a powerful framework for implementing this model. In this post, we’ll dive deep into creating an actor-based system in Rust, providing a detailed walkthrough of an implementation.

## What is the Actor Model?

The actor model treats "actors" as the primary unit of computation:
- **Encapsulation**: Each actor manages its state and behavior.
- **Asynchronous Message Passing**: Actors communicate through messages, avoiding shared mutable state.
- **Concurrency**: Each actor runs independently, providing natural parallelism.

This model is an interesting exercise for Rust due to its emphasis on safety and a great fit due to its concurrency and performance.

---

## Setting Up the Actor System

We begin with a struct representing an **Actor**, its **Handler**, and the messages it can process. We'll use **Tokio** primitives like `mpsc` channels for communication and `oneshot` channels for acknowledgments.

### Dependencies

We update `Cargo.toml` with the following dependencies:

```toml
[dependencies]
tokio = { "version" = "1.41", "features" = ["full"]}
uuid = { "version" = "1.11", "features" = ["v4", "fast-rng", "macro-diagnostics"]}
futures ={ "version" = "0.3"}
```

### Actor Messages

The `Message` enum defines the communication protocol for the actor system. Each `Message` represents an instruction sent to an actor, such as processing text (`TextMessage`), incrementing a counter (`IncrementCounter`), or stopping the actor gracefully (`Stop`). The `Stop` variant includes a `tokio::sync::oneshot::Sender` to acknowledge when the actor has fully stopped, enabling coordinated shutdowns.

```rust
// Cannot derive Clone because oneshot::Sender doesn't implement Clone
#[derive(Debug)]
enum Message {
    TextMessage {
        id: uuid::Uuid,
        body: String,
    },
    IncrementCounter {
        by: u32,
    },
    Stop {
        tx: tokio::sync::oneshot::Sender<()>,
    },
}
```

### The Actor State

The `ActorState` enum tracks the lifecycle of an actor, with states like Ready, Running, and Stopped. This state is shared between the actor and its handler (see bellow) using Arc<Mutex<RefCell<...>>> to ensure thread-safe, mutable access. This setup allows the handler to check the actor's state before sending messages or initiating a shutdown, ensuring robust control over the actor's lifecycle.

```rust
#[derive(Debug)]
enum ActorState {
    Ready,
    Running,
    Stopped,
}
```

### The Actor Structure

The `Actor` struct encapsulates:
1. **Unique Identifier**: To distinguish actors.
2. **Message Receiver**: An `mpsc::Receiver` to process incoming messages.
3. **Shared State**: Protected by `Arc<Mutex<RefCell<...>>>` for thread safety.
4. **Data**: In this case, a simple counter.

Since the proposed implementation runs both the handler and the actor in two separate thread we cannot use `RwLock` instead of `Mutex`, since it does not implement `Sync`. A `Mutex` does not distinguish between readers or writers that acquire the lock, therefore causing any tasks waiting for the lock to become available to yield. An `RwLock` will allow any number of readers to acquire the lock as long as a writer is not holding the lock.

```rust
#[derive(Debug)]
struct Actor {
    id: uuid::Uuid,
    rx: tokio::sync::mpsc::Receiver<Message>,
    state: Arc<Mutex<RefCell<ActorState>>>,
    counter: u32,
}
```

The shared state uses `Arc` for reference counting, `Mutex` for synchronization, and `RefCell` for runtime borrow-checking. These choices allow the actor's state to be modified safely across threads while retaining interior mutability.

### ActorHandler

The `ActorHandler` struct is the actor's interface, allowing external systems to send messages without directly interacting with the actor's internals.

```rust
#[derive(Debug, Clone)]
struct ActorHandler {
    id: uuid::Uuid,
    tx: tokio::sync::mpsc::Sender<Message>,
    state: Arc<Mutex<RefCell<ActorState>>>,
}
```

The `ActorHandler` holds a reference to the actor's message sender (`mpsc::Sender`) and its state. It also keeps track of the actor's state via the `state` attribute, which is a shared mutable reference (also in `Actor`).

Splitting the actor and handler is essential for ownership, concurrency, and lifecycle management. The actor owns the `Receiver` (which doesn't implement `Clone` and cannot be shared) and processes messages in isolation, while the handler owns the `Sender` (which implements `Clone`), enabling safe, thread-friendly interactions. This separation ensures the actor focuses on processing logic, while the handler provides a clean API for message passing and lifecycle control, allowing clear separation of concerns.

---

## Implementing the Actor Logic

### Handling Messages

The actor processes incoming messages via its `run` method:

```rust
async fn run(&mut self) {
    self.state.lock().unwrap().replace(ActorState::Running);

    while let Some(message) = self.rx.recv().await {
        self.process(message).await;
    }

    self.state.lock().unwrap().replace(ActorState::Stopped);
}
```

The actor continuously listens for messages using `recv`. When the channel closes, the loop ends, and the actor transitions to a `Stopped` state. Note how a mutable reference to self is passed, this is required by `recv`.

### Processing Messages

The `process` method defines how the actor handles different types of messages:

```rust
async fn process(&mut self, message: Message) {
    match message {
        Message::TextMessage { id, body } => {
            println!("Actor {} received: {}", self.id, body);
            self.counter += 1;
        }
        Message::IncrementCounter { by } => {
            self.counter += by;
        }
        Message::Stop { tx } => {
            self.state.lock().unwrap().replace(ActorState::Stopped);
            let _ = tx.send(());
            self.rx.close();
        }
    }
}
```

By matching on the `Message` enum, the actor dynamically responds to different commands. On `Message::Stop`, the actor sends an ack message to the sender.

---

## Spawning an Actor

The `ActorHandler::new` function creates an actor and spawns it on a Tokio task:

```rust
fn new(buffer: usize) -> (Self, JoinHandle<()>) {
    let (tx, rx) = tokio::sync::mpsc::channel(buffer);
    let actor_id = uuid::Uuid::new_v4();
    let state = Arc::new(Mutex::new(RefCell::new(ActorState::Ready)));
    let actor_state = state.clone();

    let actor_handler = ActorHandler { id: actor_id, tx, state };

    let handle = tokio::spawn(async move {
        let mut actor = Actor { id: actor_id, rx, state: actor_state, counter: 0 };
        actor.run().await;
    });

    (actor_handler, handle)
}
```

This function initializes the actor's components and spawns it as an asynchronous task. The handler is returned to allow for the client to join the actor's thread, i.e. finish processing messages.

---

## Communicating with the Actor

Messages are sent to the actor through the `ActorHandler::send` method:

```rust
fn send(&self, message: Message) -> JoinHandle<Result<(), tokio::sync::mpsc::error::SendError<Message>>> {
    let tx = self.tx.clone();
    tokio::spawn(async move { tx.send(message).await })
}
```

To stop an actor gracefully, we use the `ActorHandler::stop` method, which sends a `Stop` message and waits for acknowledgment:

```rust
async fn stop(&self) -> JoinHandle<Result<(), tokio::sync::mpsc::error::SendError<Message>>> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let message = Message::Stop { tx };
    let handle = self.send(message);

    let _ = rx.await; // Wait for acknowledgment
    handle
}
```

The method creates a oneshot channel for actor to acknowledge the handler that it has stopped.

---

## Key Design Considerations

1. **Concurrency**: Each actor runs independently on a Tokio task, leveraging asynchronous execution.
2. **Shared State**: The combination of `Arc<Mutex<RefCell>>` ensures thread safety and flexibility, albeit with a slight performance cost.
3. **Graceful Shutdown**: The `Stop` message and `oneshot` acknowledgment allow for a controlled shutdown process.

---

## Bringing It All Together: The `main` Function

In the main function, we create actors, send messages, and stop them.

### Defining `main` and actor instance

We start by defining main with Tokio's asynchronous runtime. We also create two actors both with a channel limit of 10:

```rust
#[tokio::main]
async fn main() {
    let (actor1, actor1_worker) = ActorHandler::new(10);
    let (actor2, actor2_worker) = ActorHandler::new(10);

    println!("[main] Actors created");
    println!("[main] Actor1: {:?}", actor1);
    println!("[main] Actor2: {:?}", actor2);
```

### Creating Messages

We then proceed to creating the messages that will be sent to both actors:

```rust
...
    // Create messages
    let messages_actor1 = vec![
        Message::TextMessage {
            id: uuid::Uuid::new_v4(),
            body: "Hello, Actor1!".to_string(),
        },
        Message::IncrementCounter {
            by: 1,
        },
    ];
    let messages_actor2 = vec![
        Message::TextMessage {
            id: uuid::Uuid::new_v4(),
            body: "Hello, Actor2!".to_string(),
        },
        Message::TextMessage {
            id: uuid::Uuid::new_v4(),
            body: "Hey again Actor2!".to_string(),
        },
        Message::IncrementCounter {
            by: 1,
        },
    ];
...
```

### Sending messages

The `send` message is called for each message, each spawning a thread. We create two vectors to store the join handles for each `send` call. This allow us to wait for the sender to finish by joining the handles.

```rust
...
    // Create vector to store the handles
    let mut actor1_handles = Vec::new();
    let mut actor2_handles = Vec::new();

    // Send messages to actor1
    for message in messages_actor1 {
        let handle = actor1.send(message);
        actor1_handles.push(handle);
    }

    // Send messages to actor2
    for message in messages_actor2 {
        let handle = actor2.send(message);
        actor2_handles.push(handle);
    }

    // Merge the two vectors
    // Cannot use `append` because `join_all` consumes an iterator
    actor1_handles.extend(actor2_handles); // inplace operation, moved value
    let handles = actor1_handles; // move value

    println!("[main] Waiting for producers to finish sending messages");
    // Wait for the producers to finish
    future::join_all(handles).await;
...
```

We merge both handle vectors for convenience to  `actor1_handles` via the `extend` method. We then move the value to `handles` for clarity.

### Stopping the actors

To gracefully shutdown the actors we call the `stop` method, that provides a correct way of sending and handling the `Message::Stop` message.

```rust
    // Stop the actors
    let stop_actor1 = actor1.stop();
    let stop_actor2 = actor2.stop();

    let _ = tokio::join!(stop_actor1, stop_actor2);

    // Check states of the actors
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await; // This is just to get different states
    println!(
        "[main] Actor1 state: '{:?}'",
        actor1.state.lock().unwrap().borrow_mut()
    );
    println!(
        "[main] Actor2 state: '{:?}'",
        actor2.state.lock().unwrap().borrow_mut()
    );
...
```

### Waiting for actors to finish processing

And finally we join the handler - await completion - to make sure that all messages have been processed before ending the program.
```rust
...
    // Wait for the actors (subscribers) to finish,
    // otherwise the program will exit before for the actors finish processing messages
    // that's because of the sleep in the actor.run() method
    // try commenting out the line below and see what happens
    // the program will exit before the actors finish processing messages
    // Can use tokio::join!, but requires passing values individually
    println!("[main] Waiting for actors to finish processing messages");
    let _ = tokio::join!(actor1_worker, actor2_worker); // Runs forever if channels are not closed

    println!("[main] Done");
    println!(
        "[main] Actor1 state: '{:?}'",
        actor1.state.lock().unwrap().borrow_mut()
    );
    println!(
        "[main] Actor2 state: '{:?}'",
        actor2.state.lock().unwrap().borrow_mut()
    );
}
```

### Example output

```bash
[ActorHandler::new] Creating new actor: af584158-9f7e-4e6c-a58f-87e263c18434
[ActorHandler::new] Creating new actor: deacb94e-90f9-44c8-bba8-44951c2029e6
[main] Actors created
[main] Actor1: ActorHandler { id: af584158-9f7e-4e6c-a58f-87e263c18434, tx: Sender { chan: Tx { inner: Chan { tx: Tx { block_tail: 0x127008600, tail_position: 0 }, semaphore: Semaphore { semaphore: Semaphore { permits: 10 }, bound: 10 }, rx_waker: AtomicWaker, tx_count: 1, rx_fields: "..." } } }, state: Mutex { data: RefCell { value: Ready }, poisoned: false, .. } }
[main] Actor2: ActorHandler { id: deacb94e-90f9-44c8-bba8-44951c2029e6, tx: Sender { chan: Tx { inner: Chan { tx: Tx { block_tail: 0x127008c00, tail_position: 0 }, semaphore: Semaphore { semaphore: Semaphore { permits: 10 }, bound: 10 }, rx_waker: AtomicWaker, tx_count: 1, rx_fields: "..." } } }, state: Mutex { data: RefCell { value: Ready }, poisoned: false, .. } }
[main] Waiting for producers to finish sending messages
[ActorHandler.stop] Stopping actor: af584158-9f7e-4e6c-a58f-87e263c18434
[ActorHandler.stop] Stopping actor: deacb94e-90f9-44c8-bba8-44951c2029e6
[Actor.run] Actor af584158-9f7e-4e6c-a58f-87e263c18434 is running
[Actor.process] Actor af584158-9f7e-4e6c-a58f-87e263c18434 received message:
	TextMessage { id: 2ca298c2-7a83-4584-9438-f05364f9bba5, body: "Hello, Actor1!" }
	counter: 1.
[Actor.process] Actor af584158-9f7e-4e6c-a58f-87e263c18434 received message: IncrementCounter { by: 1 }
	counter: 2
[Actor.process] Actor af584158-9f7e-4e6c-a58f-87e263c18434 received stop message
[Actor.run] Actor deacb94e-90f9-44c8-bba8-44951c2029e6 is running
[Actor.process] Actor deacb94e-90f9-44c8-bba8-44951c2029e6 received message:
	TextMessage { id: 2eaf4180-f6e8-406d-97bf-03b4af1192b9, body: "Hello, Actor2!" }
	counter: 1.
[Actor.run] Actor af584158-9f7e-4e6c-a58f-87e263c18434 is shutting down...
[Actor.process] Actor deacb94e-90f9-44c8-bba8-44951c2029e6 received message:
	TextMessage { id: b0aeda22-d507-4d7b-9ab6-4a3bb7cd70d2, body: "Hey again Actor2!" }
	counter: 2.
[ActorHandler.stop] Acknowledged! Actor af584158-9f7e-4e6c-a58f-87e263c18434 has stopped
[Actor.process] Actor deacb94e-90f9-44c8-bba8-44951c2029e6 received message: IncrementCounter { by: 1 }
	counter: 3
[Actor.process] Actor deacb94e-90f9-44c8-bba8-44951c2029e6 received stop message
[Actor.run] Actor deacb94e-90f9-44c8-bba8-44951c2029e6 is shutting down...
[ActorHandler.stop] Acknowledged! Actor deacb94e-90f9-44c8-bba8-44951c2029e6 has stopped
[main] Actor1 state: 'Stopped'
[main] Actor2 state: 'Stopped'
[main] Waiting for actors to finish processing messages
[main] Done
[main] Actor1 state: 'Stopped'
[main] Actor2 state: 'Stopped'
```

---

## Conclusion

This implementation highlights the intricacies of building an actor model in Rust, leveraging its ownership and concurrency guarantees. By separating the actor and handler, we achieve a clean design that supports message-driven communication, lifecycle management, and concurrency. Rust's strict ownership rules need careful consideration, such as using `Arc<Mutex<RefCell<...>>>` for shared state and ensuring that non-clonable components like `Receiver` remain isolated within the actor. This design showcases the power and challenges of combining Rust's safety with asynchronous programming, offering a robust foundation for building scalable, concurrent systems.

## References

- [actors-with-tokio](https://ryhl.io/blog/actors-with-tokio/)
- [Actor Model Explained](https://www.youtube.com/watch?v=ELwEdb_pD0k)
- [Tokio RwLock](https://docs.rs/tokio/latest/tokio/sync/struct.RwLock.html)

## Appendix

<details>

<summary> The Whole Program </summary>

You can find thw hole implementation bellow:

```rust
    use futures::future;
    use std::cell::RefCell;
    use std::sync::{Arc, Mutex};
    use tokio::{self, task::JoinHandle};

    use uuid;

    #[derive(Debug)] // Cannot derive Clone because oneshot::Sender doesn't implement Clone
    enum Message {
        TextMessage {
            id: uuid::Uuid,
            body: String,
        },
        IncrementCounter {
            by: u32,
        },
        Stop {
            tx: tokio::sync::oneshot::Sender<()>,
        },
    }

    #[derive(Debug)]
    enum ActorState {
        Ready,
        Running,
        Stopped,
    }

    #[derive(Debug)]
    struct Actor {
        // Responsible for processing data
        // Should run in its own thread in the background
        id: uuid::Uuid,
        rx: tokio::sync::mpsc::Receiver<Message>, // Can't derive `Clone` for `mpsc::Receiver`


        state: Arc<Mutex<RefCell<ActorState>>>,   // Requires `Mutex` to be shared between threads
        // Cannot use RwLock since it's not Sync and cannot be safely shared among threads.
        // state: Arc<tokio::sync::RwLock<RefCell<ActorState>>>

        // Actor's data
        counter: u32,
    }

    #[derive(Debug, Clone)]
    struct ActorHandler {
        // Hard-linked to an actor, which doe the actual computation
        id: uuid::Uuid,
        tx: tokio::sync::mpsc::Sender<Message>,
        state: Arc<Mutex<RefCell<ActorState>>>, // Needs Arc<Mutex<RefCell<...>>> because state is shared with `Actor`
                                                // Can't derive `Clone` for `JoinHandle`
                                                // My idea was to store the handle here so that
                                                // we could wait for the actor to finish processing messages
                                                // -> handle: Option<JoinHandle<()>>,
    }

    impl Actor {
        async fn run(&mut self) {
            println!("[Actor.run] Actor {} is running", self.id);
            // Can use `RefCell`'s `replace` to change the state of the actor
            self.state.lock().unwrap().replace(ActorState::Running);

            // Consume messages while the channel is open
            while let Some(message) = self.rx.recv().await {
                self.process(message).await;
            }

            self.state.lock().unwrap().replace(ActorState::Stopped);

            println!("[Actor.run] Actor {} is shutting down...", self.id);
        }

        async fn process(&mut self, message: Message) {
            match message {
                Message::TextMessage { id, body } => {
                    println!(
                        "[Actor.process] Actor {} received message:\n\t{:?}",
                        self.id, Message::TextMessage { id, body }
                    );
                    self.counter += 1;
                    println!("\tcounter: {}.", self.counter);
                }
                Message::IncrementCounter { by } => {
                    self.counter += by;
                    println!("[Actor.process] Actor {} received message: {:?}", self.id, Message::IncrementCounter { by });
                    println!("\tcounter: {}", self.counter);
                }
                Message::Stop { tx } => {
                    println!("[Actor.process] Actor {} received stop message", self.id);
                    self.state.lock().unwrap().replace(ActorState::Stopped);

                    // Acknowledge handler that actor has stopped.
                    // Since this is a oneshot channel it gets dropped right away
                    let _ = tx.send(());

                    // Close actor's channel
                    self.rx.close();
                }
            }
        }
    }

    impl ActorHandler {

        // Creates an Actor and starts it in a separate thread.
        // Returns an ActorHandler instance and a thread handle,
        // which belong to the Actor which is consuming messages.
        // This allows waiting for the actor to finish processing all the messages.
        fn new(buffer: usize) -> (Self, JoinHandle<()>) {
            let (tx, rx) = tokio::sync::mpsc::channel(buffer);
            let actor_id = uuid::Uuid::new_v4();
            let state = Arc::new(Mutex::new(RefCell::new(ActorState::Ready)));
            let actor_state = state.clone();

            println!("[ActorHandler::new] Creating new actor: {}", actor_id);

            let actor_handler = ActorHandler {
                id: actor_id,
                tx: tx,
                state: state,
            };

            // Can use std::thread::spawn instead if we want to run the actor in a dedicated thread.
            // Can come handy when `process` does blocking IO.
            let handle = tokio::spawn(async move {
                let mut actor = Actor {
                    id: actor_id,
                    rx: rx,
                    state: actor_state,
                    counter: 0,
                };
                // yield the control to the tokio runtime (for illustrative purposes, makes output more _unordered_)
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

                actor.run().await;
            });

            (actor_handler, handle)
        }

        fn send(
            &self,
            message: Message,
        ) -> JoinHandle<Result<(), tokio::sync::mpsc::error::SendError<Message>>> {
            // Validate that the actor is active.
            match *self.state.lock().unwrap().borrow_mut() {
                ActorState::Stopped => {
                    println!("[ActorHandler::send] Actor {} is stopped", self.id);
                    return tokio::spawn(async { Err(tokio::sync::mpsc::error::SendError(message)) });
                }
                _ => {}
            }

            let tx = self.tx.clone();

            // Send message on separate thread.
            let handle = tokio::spawn(async move {
                tx.send(message).await
            });

            // Return handle so that thread can be joined.
            handle
        }

        // Stop the actor.
        // This is similar to the `send` method, but does not require passing an instance
        // of `Message::stop` nor for that method to process the ack or for the ack to be processed somewhere else.
        async fn stop(&self) -> JoinHandle<Result<(), tokio::sync::mpsc::error::SendError<Message>>> {
            let (tx, rx) = tokio::sync::oneshot::channel::<()>();
            let message = Message::Stop { tx: tx };

            match *self.state.lock().unwrap().borrow_mut(){
                ActorState::Stopped => {
                    println!("[ActorHandler.stop] Actor {} is stopped", self.id);
                    return tokio::spawn(async { Err(tokio::sync::mpsc::error::SendError(message)) });
                }
                _ => {}
            }

            println!("[ActorHandler.stop] Stopping actor: {}", self.id);

            let handle = self.send(message);

            // Block actor. Can spawn a thread for non blocking behavior.
            let _ = rx.await;

            println!("[ActorHandler.stop] Acknowledged! Actor {} has stopped", self.id);

            handle
        }
    }

    #[tokio::main]
    async fn main() {
        let (actor1, actor1_worker) = ActorHandler::new(10);
        let (actor2, actor2_worker) = ActorHandler::new(10);

        println!("[main] Actors created");
        println!("[main] Actor1: {:?}", actor1);
        println!("[main] Actor2: {:?}", actor2);

        // Create messages
        let messages_actor1 = vec![
            Message::TextMessage {
                id: uuid::Uuid::new_v4(),
                body: "Hello, Actor1!".to_string(),
            },
            Message::IncrementCounter {
                by: 1,
            },
        ];
        let messages_actor2 = vec![
            Message::TextMessage {
                id: uuid::Uuid::new_v4(),
                body: "Hello, Actor2!".to_string(),
            },
            Message::TextMessage {
                id: uuid::Uuid::new_v4(),
                body: "Hey again Actor2!".to_string(),
            },
            Message::IncrementCounter {
                by: 1,
            },
        ];

        // Create vector to store the handles
        let mut actor1_handles = Vec::new();
        let mut actor2_handles = Vec::new();

        // Send messages to actor1
        for message in messages_actor1 {
            let handle = actor1.send(message);
            actor1_handles.push(handle);
        }

        // Send messages to actor2
        for message in messages_actor2 {
            let handle = actor2.send(message);
            actor2_handles.push(handle);
        }

        // Merge the two vectors
        // Cannot use `append` because `join_all` consumes an iterator
        actor1_handles.extend(actor2_handles); // inplace operation, moved value
        let handles = actor1_handles; // move value

        println!("[main] Waiting for producers to finish sending messages");
        // Wait for the producers to finish
        future::join_all(handles).await;

        // Stop the actors
        let stop_actor1 = actor1.stop();
        let stop_actor2 = actor2.stop();

        let _ = tokio::join!(stop_actor1, stop_actor2);

        // Check states of the actors and if channels are closed
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        println!(
            "[main] Actor1 state: '{:?}'",
            actor1.state.lock().unwrap().borrow_mut()
        );
        println!(
            "[main] Actor2 state: '{:?}'",
            actor2.state.lock().unwrap().borrow_mut()
        );

        // Wait for the actors (subscribers) to finish,
        // otherwise the program will exit before for the actors finish processing messages
        // that's because of the sleep in the actor.run() method
        // try commenting out the line below and see what happens
        // the program will exit before the actors finish processing messages
        //
        // Can use tokio::join!, but requires passing values individually
        println!("[main] Waiting for actors to finish processing messages");
        let _ = tokio::join!(actor1_worker, actor2_worker); // Runs forever if channels are not closed

        println!("[main] Done");
        println!(
            "[main] Actor1 state: '{:?}'",
            actor1.state.lock().unwrap().borrow_mut()
        );
        println!(
            "[main] Actor2 state: '{:?}'",
            actor2.state.lock().unwrap().borrow_mut()
        );
    }
```

</details>
