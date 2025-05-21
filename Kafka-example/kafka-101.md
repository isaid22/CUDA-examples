## Step 1: Prerequisites

It is recommended to use [Terraform example](../Terraform/t3/) to provion an instance to satisfy these prerequisites.
1. Install Java (JDK 11+)

```
sudo apt update
sudo apt install openjdk-21-jdk -y
java -version  # Verify installation
```

2. Download Kafka


Get the binary 

```
wget https://downloads.apache.org/kafka/3.7.2/kafka_2.13-3.7.2.tgz 
tar -xzf kafka_2.13-3.7.2.tgz
cd kafka_2.13-3.7.2

```

## Step 2: Start Kafka Services

1. Start ZooKeeper (required for Kafka <3.0; newer versions can use KRaft mode, but we’ll use ZooKeeper for learning):

```
bin/zookeeper-server-start.sh config/zookeeper.properties
```

2. Start Kafka Broker (in a new terminal):

```
bin/kafka-server-start.sh config/server.properties
```

## Step 3: Create a Topic

Open a third terminal and create a test topic:

```
bin/kafka-topics.sh --create --topic quickstart-events --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

Verify the topic:

```
bin/kafka-topics.sh --describe --topic quickstart-events --bootstrap-server localhost:9092
```

Step 4: Produce and COnsume Messages

1. Start a Producer (write messages):

```
bin/kafka-console-producer.sh --topic quickstart-events --bootstrap-server localhost:9092
```

Type a few messages (e.g., Hello, Kafka!).

2. Start a Consumer (read messages in a new terminal):

```
bin/kafka-console-consumer.sh --topic quickstart-events --from-beginning --bootstrap-server localhost:9092
```

You’ll see the messages you typed earlier.


## Final Step: Teardown

Stop Kafka and ZooKeeper with `Ctrl+C` in their terminals. Delete logs:

```
rm -rf /tmp/kafka-logs /tmp/zookeeper
```