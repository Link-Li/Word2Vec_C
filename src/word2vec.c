//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#include <time.h>
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers 浮点数精度

// 词汇中每个单词对应的结构体
struct vocab_word
{
    long long cn; // 单词出现的次数
    int *point;   // 保存单词路径中的每个节点的index值，包括了根节点
    char *word; // 单词的内容
    char *code;  // 保存单词路径中每次是走左边还是走右边，不包括根节点
    char codelen;  // 单词的哈夫曼编码的长度，不包括根节点
};

// 训练源文件和输出文件
char train_file[MAX_STRING], output_file[MAX_STRING];
// 保存词汇文件和读词汇文件
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

/*
初始化负采样概率表
*/
void InitUnigramTable()
{
    int a, i;
    double train_words_pow = 0;
    double d1, power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
    i = 0;
    d1 = pow(vocab[i].cn, power) / train_words_pow;
    for (a = 0; a < table_size; a++)
    {
        table[a] = i;
        if (a / (double)table_size > d1)
        {
            i++;
            d1 += pow(vocab[i].cn, power) / train_words_pow;
        }
        if (i >= vocab_size) i = vocab_size - 1;
    }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
// 从文件中读取单个单词，每个单词中通过空格或者tab或者EOL进行分割
void ReadWord(char *word, FILE *fin)
{
    int a = 0, ch;
    // 当设置了与流关联的文件结束标识符时，该函数返回一个非零值，否则返回零。
    // 就是读完了文件之后退出循环
    while (!feof(fin))
    {
        ch = fgetc(fin);
        // 表示回车
        if (ch == 13)
            continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
        {
            if (a > 0)
            {
                if (ch == '\n')
                    ungetc(ch, fin);
                break;
            }
            if (ch == '\n')
            {
                strcpy(word, (char *)"</s>");
                return;
            }
            else
                continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1)
            a--;   // Truncate too long words
    }
    word[a] = 0;
}

// Returns hash value of a word
// 返回单词对应的哈希值
int GetWordHash(char *word)
{
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++)
        hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
// 返回单词在词汇表中的位置，如果单词没有找到就返回-1
// 线性探测的开放定址法
int SearchVocab(char *word)
{
    unsigned int hash = GetWordHash(word);
    while (1)
    {
        if (vocab_hash[hash] == -1)
            return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word))
            return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Reads a word and returns its index in the vocabulary
// 读取单词在词汇表中的索引
int ReadWordIndex(FILE *fin)
{
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    return SearchVocab(word);
}

// Adds a word to the vocabulary
// 添加一个单词到词汇表中
int AddWordToVocab(char *word)
{
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING)
        length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size)
    {
        vocab_max_size += 1000;
        // 重新调整之前调用 malloc 或 calloc 所分配的 ptr 所指向的内存块的大小。
        vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1)
        hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

// Used later for sorting by word counts
// 构造一个比较器，用于一会单词的排序
int VocabCompare(const void *a, const void *b)
{
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
// 通过词频对单词进行排序
void SortVocab()
{
    int a, size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    // 排序，并且保持</s>在第一个位置不变动
    // 这里的排序很重要，因为这里是按照从大到小进行的排序，所以下面在释放内存，也就是删去小于min_count的词的时候
    // 实际上是从vocab这个词汇表的结尾进行逐个删除的，所以后面进行realloc的时候，实际上是将已经释放内存的那部分删去了
    // 所以这样并没有影响没被删去的词
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    for (a = 0; a < vocab_hash_size; a++)
        vocab_hash[a] = -1;
    size = vocab_size;
    train_words = 0;
    for (a = 0; a < size; a++)
    {
        // Words occuring less than min_count times will be discarded from the vocab
        // 当词频小于min_count的时候会被丢弃
        if ((vocab[a].cn < min_count) && (a != 0))
        {
            vocab_size--;
            free(vocab[a].word);
        }
        else
        {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetWordHash(vocab[a].word);
            while (vocab_hash[hash] != -1)
                hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = a;
            train_words += vocab[a].cn;
        }
    }
    vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
    // Allocate memory for the binary tree construction
    // 为构建哈弗曼树而分配内存
    for (a = 0; a < vocab_size; a++)
    {
        vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    }
}

// Reduces the vocabulary by removing infrequent tokens
// 删除词典中的不经常出现的词
void ReduceVocab()
{
    int a, b = 0;
    unsigned int hash;
    for (a = 0; a < vocab_size; a++)
        if (vocab[a].cn > min_reduce)
        {
            vocab[b].cn = vocab[a].cn;
            vocab[b].word = vocab[a].word;
            b++;
        }
        else
            free(vocab[a].word);
    vocab_size = b;
    for (a = 0; a < vocab_hash_size; a++)
        vocab_hash[a] = -1;
    for (a = 0; a < vocab_size; a++)
    {
        // Hash will be re-computed, as it is not actual
        hash = GetWordHash(vocab[a].word);
        while (vocab_hash[hash] != -1)
            hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = a;
    }
    fflush(stdout);
    // 每次裁剪之后都会增加最低频率数
    min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
// 通过词频，构建了一个单词的哈夫曼二叉树，频率较高的词将会有一个比较短的二进制编码
void CreateBinaryTree()
{
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];
    // 分配的空间大小为，(vocab_size * 2 + 1) * sizeof(long long),因为hufuman树的特性，
    // 所以总结点数是2 * n + 1, 其中n是单词的数量, 此处应该有错误，是2 * n - 1才对
    long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));   // 节点对应的频率
    long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));  // 记录每个几点是左节点还是右节点
    long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));  // 父节点位置
    for (a = 0; a < vocab_size; a++)
        count[a] = vocab[a].cn;  // count的值是从大到小的
    //for(a = 0; a < 10; a++)
    //printf("%ld\n", count[a]);
    for (a = vocab_size; a < vocab_size * 2; a++)
        count[a] = 1e15;  // 设置为无穷大，这里是为了下面构造父节点
    // 这个是用于读取单词的index，从后向前，也就是从单词频率小的到单词频率大的
    pos1 = vocab_size - 1;
    // 这个是用于读取父节点的index，从前向后
    pos2 = vocab_size;
    // Following algorithm constructs the Huffman tree by adding one node at a time
    for (a = 0; a < vocab_size - 1; a++)
    {
        // First, find two smallest nodes 'min1, min2'
        if (pos1 >= 0)
        {
            if (count[pos1] < count[pos2])
            {
                min1i = pos1;
                pos1--;
            }
            else
            {
                min1i = pos2;
                pos2++;
            }
        }
        else
        {
            min1i = pos2;
            pos2++;
        }
        if (pos1 >= 0)
        {
            if (count[pos1] < count[pos2])
            {
                min2i = pos1;
                pos1--;
            }
            else
            {
                min2i = pos2;
                pos2++;
            }
        }
        else
        {
            min2i = pos2;
            pos2++;
        }
        count[vocab_size + a] = count[min1i] + count[min2i];
        parent_node[min1i] = vocab_size + a;
        parent_node[min2i] = vocab_size + a;
        binary[min2i] = 1;
    }
    // Now assign binary code to each vocabulary word
    for (a = 0; a < vocab_size; a++)
    {
        b = a;
        i = 0;
        while (1)
        {
            code[i] = binary[b];    // 对于每个单词节点，从底向上获取得到code值，上面是使用了binary来记录的该节点是左节点还是右节点
            point[i] = b;                  // 用于记录单词节点到根节点的路径
            i++;
            b = parent_node[b];  // 因为是哈弗曼树，所以单词节点不会是任何一个子树的父节点，所以根据这个父节点就可以获取到单词节点到根节点的路径
            if (b == vocab_size * 2 - 2)  // 说明已经到了根节点
                break;
        }
        vocab[a].codelen = i;  // 记录了每个单词对应的哈夫曼编码的长度
        vocab[a].point[0] = vocab_size - 2;  // 将根节点的index作为路径的第一个索引
        for (b = 0; b < i; b++)
        {
            vocab[a].code[i - b - 1] = code[b];   // 这里进行了倒序存储，自顶向下
            // 这个索引对应的是哈夫曼树的非叶子节点，对应的是syn1的索引
            // 因为考虑到syn1是从0开始的，而不是从vocab_size开始的，所以这个地方需要在原来的基础上面减去vocab_size
            vocab[a].point[i - b] = point[b] - vocab_size;
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

// 整合上面的操作
void LearnVocabFromTrainFile()
{
    char word[MAX_STRING];
    FILE *fin;
    long long a, i;
    for (a = 0; a < vocab_hash_size; a++)
        vocab_hash[a] = -1;
    fin = fopen(train_file, "rb");
    if (fin == NULL)
    {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    vocab_size = 0;
    AddWordToVocab((char *)"</s>");
    while (1)
    {
        ReadWord(word, fin);
        if (feof(fin))
            break;
        train_words++;
        if ((debug_mode > 1) && (train_words % 100000 == 0))
        {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        i = SearchVocab(word);
        if (i == -1)
        {
            a = AddWordToVocab(word);
            vocab[a].cn = 1;
        }
        else
            vocab[i].cn++;
        // 当前词典规模满足vocab_size > vocab_hash_size * 0.7的时候，就会将词典中小于min_reduce的词删除
        if (vocab_size > vocab_hash_size * 0.7)
            ReduceVocab();
    }
    SortVocab();
    if (debug_mode > 0)
    {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
        printf("%lldK%c\n", train_words / 1000, 13);
    }
    // 该函数返回位置标识符的当前值,这里因为是读取文件到了结尾，相当于返回了文件的大小，单位是字节
    file_size = ftell(fin);
    fclose(fin);
}

// 保存学习到的词汇文件表
void SaveVocab()
{
    long long i;
    FILE *fo = fopen(save_vocab_file, "wb");
    for (i = 0; i < vocab_size; i++)
        fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}


void ReadVocab()
{
    long long a, i = 0;
    char c;
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL)
    {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < vocab_hash_size; a++)
        vocab_hash[a] = -1;
    vocab_size = 0;
    while (1)
    {
        ReadWord(word, fin);
        if (feof(fin))
            break;
        a = AddWordToVocab(word);
        fscanf(fin, "%lld%c", &vocab[a].cn, &c);
        i++;
    }
    SortVocab();
    if (debug_mode > 0)
    {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    fin = fopen(train_file, "rb");
    if (fin == NULL)
    {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_END);
    file_size = ftell(fin);
    fclose(fin);
}

// 初始化网络
void InitNet()
{
    long long a, b;
    unsigned long long next_random = 1;

#ifdef _MSC_VER
    syn0 = _aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
#elif defined  linux
    // 为syn0分配内存，对齐的内存，大小为vocab_size * layer1_size * sizeof(real),也就是每个词汇对应一个layer1_size的向量
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
#endif

    if (syn0 == NULL)
    {
        printf("Memory allocation failed\n");
        exit(1);
    }
    if (hs)
    {
#ifdef _MSC_VER
        syn1 = _aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
#elif defined  linux
        // 如果采用huffman softmax构造，那么需要初始化syn1，大小为vocab_size * layer1_size * sizeof(real)，每个词对应一个
        a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
#endif
        if (syn1 == NULL)
        {
            printf("Memory allocation failed\n");
            exit(1);
        }
        for (a = 0; a < vocab_size; a++)
            for (b = 0; b < layer1_size; b++)
                syn1[a * layer1_size + b] = 0;
    }
    if (negative>0)
    {
#ifdef _MSC_VER
        syn1neg = _aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
#elif defined  linux
        a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
#endif
        //如果采用负采样进行训练，那么久初始化syn1neg，大小为vocab_size * layer1_size * sizeof(real)，每个词对应一个
        if (syn1neg == NULL)
        {
            printf("Memory allocation failed\n");
            exit(1);
        }
        for (a = 0; a < vocab_size; a++)
            for (b = 0; b < layer1_size; b++)
                syn1neg[a * layer1_size + b] = 0;
    }

    //对syn0中每个词对应的词向量进行初始化
    // 生产伪随机数，这里和C++的内部实现类似，但是这里的随机数种子一直都是1，没有改变而已
    // 参考 https://www.cnblogs.com/xkfz007/archive/2012/08/25/2656893.html
    // 这个代码中的所有的随机变量生产的数据的范围都是[0, 1],这里是因为减了一个0.5，除了一个layer1_size
    for (a = 0; a < vocab_size; a++)
        for (b = 0; b < layer1_size; b++)
        {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
            // double p = (((next_random & 0xFFFF) / (real)65536) - 0.5);
            // int pp = 1;
        }
    CreateBinaryTree();
}

/*
首先，这里虽然使用了多线程优化，但是并没有对线程中的梯度更新等做加锁。
一般来说，加锁之后效果肯定会更好，因为多线程必定会有同时对某一个词向量的某一个数据进行更新。
但是从网上查资料，
https://www.zhihu.com/question/29273081/answer/43805236
https://www.zhihu.com/question/29273081/answer/43777828
以及我自己思考，加锁之后肯定会影响效率，而且影响巨大，因为主要的计算操作（涉及到syn1，syn0这样的全局变量的计算）都需要加锁。
本身为了提高计算速度，使用多线程，就是希望可以同时对这些数据进行计算，加上锁之后也就没有什么意义了。
同时要考虑到发生冲突的概率，一般而言，一个词向量长度为200，词的数量超出1万，syn1和syn0的大小就在百万以上了。
这个时候即使有20个线程在运行，假设其中两个线程发生了冲突，那么发生的概率也只有(一百万分之一)^2。
所以即使加上锁，效果提升的也比较少，但是效率下降的很厉害
*/
// 训练模型的线程
void *TrainModelThread(void *id)
{
    long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label, local_iter = iter;
    unsigned long long next_random = (long long)id;
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(train_file, "rb");
    // SEEK_SET： 文件开头
    fseek(fi, (file_size / (long long)num_threads) * (long long)id, SEEK_SET);
    while (1)
    {
        if (word_count - last_word_count > 10000)
        {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1))
            {
                now=clock();
                // CLOCKS_PER_SEC是标准c的time.h头函数中宏定义的一个常数，
                // 表示一秒钟内CPU运行的时钟周期数，用于将clock()函数的结果转化为以秒为单位的量，
                // 但是这个量的具体值是与操作系统相关的。
                printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  Word_count_actual: %ld", 13, alpha,
                       word_count_actual / (real)(iter * train_words + 1) * 100,
                       word_count_actual / (((real)(now - start + 1) / (real)CLOCKS_PER_SEC) * 1000), word_count_actual);
                fflush(stdout);
            }
            alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
            if (alpha < starting_alpha * 0.0001)
                alpha = starting_alpha * 0.0001;
        }
        if (sentence_length == 0)
        {
            while (1)
            {
                word = ReadWordIndex(fi);
                if (feof(fi))
                    break;
                if (word == -1)
                    continue;
                word_count++;
                // 如果是回车，那么就停止继续获取单词
                if (word == 0)
                    break;
                // The subsampling randomly discards frequent words while keeping the ranking same
                // 下采样随机丢弃频繁的单词，同时保持单词的排名不变，随机的跳过一些单词的训练
                if (sample > 0)
                {
                    real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / (real)65536)
                        continue;
                }
                sen[sentence_length] = word;
                sentence_length++;
                // 这个训练的语料有点长，这个好像是一个文章，所以整个下来超过了1000个单词，那么就强行截断
                if (sentence_length >= MAX_SENTENCE_LENGTH)
                    break;
            }
            sentence_position = 0;
        }
        if (feof(fi) || (word_count > train_words / num_threads))
        {
            // 进行重置一些变量，用于下一次迭代做准备
            word_count_actual += word_count - last_word_count;
            local_iter--;
            // 迭代结束，退出循环
            if (local_iter == 0)
                break;
            word_count = 0;
            last_word_count = 0;
            sentence_length = 0;
            fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
            continue;
        }
        word = sen[sentence_position];
        if (word == -1)
            continue;
        for (c = 0; c < layer1_size; c++)
            neu1[c] = 0;
        for (c = 0; c < layer1_size; c++)
            neu1e[c] = 0;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        // 用随机数在区间[1，Windows]上来生成一个窗口的大小
        b = next_random % window;
        if (cbow)    //train the cbow architecture
        {
            // in -> hidden
            cw = 0;
            for (a = b; a < window * 2 + 1 - b; a++)
                if (a != window)
                {
                    c = sentence_position - window + a;
                    if (c < 0)
                        continue;
                    if (c >= sentence_length)
                        continue;
                    last_word = sen[c];
                    if (last_word == -1)
                        continue;
                    // 投影层，将单词的向量syn0投影到neul中
                    for (c = 0; c < layer1_size; c++)
                        neu1[c] += syn0[c + last_word * layer1_size];
                    cw++;
                }
            if (cw)
            {
                for (c = 0; c < layer1_size; c++)
                    neu1[c] /= cw;
                if (hs)
                    for (d = 0; d < vocab[word].codelen; d++)
                    {
                        f = 0;
                        l2 = vocab[word].point[d] * layer1_size;
                        // Propagate hidden -> output
                        // 因为这里使用syn1是一维的数组，所以需要使用l2来作为偏移量来获取到对应单词对应的词向量
                        for (c = 0; c < layer1_size; c++)
                            f += neu1[c] * syn1[c + l2];
                        // 当所得到的f不在[-6,6]的范围的时候，就可以舍弃，因为超过这个范围之后，sigmoid进行反向传播的时候几乎为0
                        if (f <= -MAX_EXP)
                            continue;
                        else
                            if (f >= MAX_EXP)
                                continue;
                        else
                            f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        // 'g' is the gradient multiplied by the learning rate
                        g = (1 - vocab[word].code[d] - f) * alpha;
                        // Propagate errors output -> hidden
                        for (c = 0; c < layer1_size; c++)
                            neu1e[c] += g * syn1[c + l2];
                        // Learn weights hidden -> output
                        for (c = 0; c < layer1_size; c++)
                            syn1[c + l2] += g * neu1[c];
                    }
                // NEGATIVE SAMPLING
                if (negative > 0)
                    for (d = 0; d < negative + 1; d++)
                    {
                        if (d == 0)
                        {
                            target = word;
                            label = 1;
                        }
                        else
                        {
                            next_random = next_random * (unsigned long long)25214903917 + 11;
                            target = table[(next_random >> 16) % table_size];
                            if (target == 0)
                                target = next_random % (vocab_size - 1) + 1;
                            if (target == word)
                                continue;
                            label = 0;
                        }
                        l2 = target * layer1_size;
                        f = 0;
                        for (c = 0; c < layer1_size; c++)
                            f += neu1[c] * syn1neg[c + l2];
                        if (f > MAX_EXP)
                            g = (label - 1) * alpha;
                        else
                            if (f < -MAX_EXP)
                                g = (label - 0) * alpha;
                        else
                            g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                        for (c = 0; c < layer1_size; c++)
                            neu1e[c] += g * syn1neg[c + l2];
                        for (c = 0; c < layer1_size; c++)
                            syn1neg[c + l2] += g * neu1[c];
                    }
                // hidden -> in
                for (a = b; a < window * 2 + 1 - b; a++) if (a != window)
                    {
                        c = sentence_position - window + a;
                        if (c < 0)
                            continue;
                        if (c >= sentence_length)
                            continue;
                        last_word = sen[c];
                        if (last_word == -1)
                            continue;
                        for (c = 0; c < layer1_size; c++)
                            syn0[c + last_word * layer1_size] += neu1e[c];
                    }
            }
        }
        else      //train skip-gram
        {
            for (a = b; a < window * 2 + 1 - b; a++) if (a != window)
                {
                    c = sentence_position - window + a;
                    if (c < 0)
                        continue;
                    if (c >= sentence_length)
                        continue;
                    last_word = sen[c];
                    if (last_word == -1)
                        continue;
                    l1 = last_word * layer1_size;
                    for (c = 0; c < layer1_size; c++)
                        neu1e[c] = 0;
                    // HIERARCHICAL SOFTMAX
                    if (hs)
                        for (d = 0; d < vocab[word].codelen; d++)
                        {
                            f = 0;
                            l2 = vocab[word].point[d] * layer1_size;
                            // Propagate hidden -> output
                            for (c = 0; c < layer1_size; c++)
                                f += syn0[c + l1] * syn1[c + l2];
                            if (f <= -MAX_EXP)
                                continue;
                            else if (f >= MAX_EXP)
                                continue;
                            else
                                f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                            // 'g' is the gradient multiplied by the learning rate
                            g = (1 - vocab[word].code[d] - f) * alpha;
                            // Propagate errors output -> hidden
                            for (c = 0; c < layer1_size; c++)
                                neu1e[c] += g * syn1[c + l2];
                            // Learn weights hidden -> output
                            for (c = 0; c < layer1_size; c++)
                                syn1[c + l2] += g * syn0[c + l1];
                        }
                    // NEGATIVE SAMPLING
                    if (negative > 0)
                        for (d = 0; d < negative + 1; d++)
                        {
                            if (d == 0)
                            {
                                target = word;
                                label = 1;
                            }
                            else
                            {
                                next_random = next_random * (unsigned long long)25214903917 + 11;
                                target = table[(next_random >> 16) % table_size];
                                if (target == 0)
                                    target = next_random % (vocab_size - 1) + 1;
                                if (target == word)
                                    continue;
                                label = 0;
                            }
                            l2 = target * layer1_size;
                            f = 0;
                            for (c = 0; c < layer1_size; c++)
                                f += syn0[c + l1] * syn1neg[c + l2];
                            if (f > MAX_EXP)
                                g = (label - 1) * alpha;
                            else if (f < -MAX_EXP)
                                g = (label - 0) * alpha;
                            else
                                g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                            for (c = 0; c < layer1_size; c++)
                                neu1e[c] += g * syn1neg[c + l2];
                            for (c = 0; c < layer1_size; c++)
                                syn1neg[c + l2] += g * syn0[c + l1];
                        }
                    // Learn weights input -> hidden
                    for (c = 0; c < layer1_size; c++)
                        syn0[c + l1] += neu1e[c];
                }
        }
        sentence_position++;
        if (sentence_position >= sentence_length)
        {
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
#ifdef _MSC_VER
    _endthreadex(0);
#elif defined  linux
    pthread_exit(NULL);
#endif
}

#ifdef _MSC_VER
DWORD WINAPI TrainModelThread_win(LPVOID tid)
{
    TrainModelThread(tid);
    return 0;
}
#endif

void TrainModel()
{
    long a, b, c, d;
    FILE *fo;
    printf("Starting training using file %s\n", train_file);
    starting_alpha = alpha;
    if (read_vocab_file[0] != 0)
        ReadVocab();
    else
        LearnVocabFromTrainFile();
    if (save_vocab_file[0] != 0)
        SaveVocab();
    if (output_file[0] == 0)
        return;
    InitNet();
    if (negative > 0)
        InitUnigramTable();
    start = clock();

#ifdef _MSC_VER
    HANDLE *pt = (HANDLE *)malloc(num_threads * sizeof(HANDLE));
    for (int i = 0; i < num_threads; i++)
    {
        pt[i] = (HANDLE)_beginthreadex(NULL, 0, TrainModelThread_win, (void *)i, 0, NULL);
    }
    WaitForMultipleObjects(num_threads, pt, TRUE, INFINITE);
    for (int i = 0; i < num_threads; i++)
    {
        CloseHandle(pt[i]);
    }
    free(pt);
#elif defined  linux
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    for (a = 0; a < num_threads; a++)
        pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++)
        pthread_join(pt[a], NULL);
#endif

    fo = fopen(output_file, "wb");
    if (classes == 0)
    {
        // Save the word vectors
        fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
        for (a = 0; a < vocab_size; a++)
        {
            fprintf(fo, "%s ", vocab[a].word);
            if (binary)
                for (b = 0; b < layer1_size; b++)
                    fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
            else
                for (b = 0; b < layer1_size; b++)
                    fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
            fprintf(fo, "\n");
        }
    }
    else
    {
        // Run K-means on the word vectors
        int clcn = classes, iter = 10, closeid;
        int *centcn = (int *)malloc(classes * sizeof(int));
        int *cl = (int *)calloc(vocab_size, sizeof(int));
        real closev, x;
        real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
        for (a = 0; a < vocab_size; a++)
            cl[a] = a % clcn;
        for (a = 0; a < iter; a++)
        {
            for (b = 0; b < clcn * layer1_size; b++)
                cent[b] = 0;
            for (b = 0; b < clcn; b++)
                centcn[b] = 1;
            for (c = 0; c < vocab_size; c++)
            {
                for (d = 0; d < layer1_size; d++)
                    cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
                centcn[cl[c]]++;
            }
            for (b = 0; b < clcn; b++)
            {
                closev = 0;
                for (c = 0; c < layer1_size; c++)
                {
                    cent[layer1_size * b + c] /= centcn[b];
                    closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
                }
                closev = sqrt(closev);
                for (c = 0; c < layer1_size; c++)
                    cent[layer1_size * b + c] /= closev;
            }
            for (c = 0; c < vocab_size; c++)
            {
                closev = -10;
                closeid = 0;
                for (d = 0; d < clcn; d++)
                {
                    x = 0;
                    for (b = 0; b < layer1_size; b++)
                        x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
                    if (x > closev)
                    {
                        closev = x;
                        closeid = d;
                    }
                }
                cl[c] = closeid;
            }
        }
        // Save the K-means classes
        for (a = 0; a < vocab_size; a++)
            fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
        free(centcn);
        free(cent);
        free(cl);
    }
    fclose(fo);
}

int ArgPos(char *str, int argc, char **argv)
{
    int a;
    for (a = 1; a < argc; a++)
        if (!strcmp(str, argv[a]))
        {
            if (a == argc - 1)
            {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    return -1;
}

/*
argc 参数的个数，因为C语言和Python不太一样，所以这里的参数个数是实际输入参数的（2倍+1）
*/
int main(int argc, char **argv)
{
    //char buf[80];
    // getcwd(buf,sizeof(buf));
    // printf("current working directory: %s\n", buf);
    int i;
    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;
    if (argc == 1)
    {
        printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
        printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-hs <int>\n");
        printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tRun more training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
        printf("\t-classes <int>\n");
        printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
        printf("\t-cbow <int>\n");
        printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");

        // 调试方便，直接把参数写到了这里
        strcpy(train_file, "data/text8");
        strcpy(output_file, "data/text8-vector-test.bin");
        strcpy(read_vocab_file, "data/text8-vocab.bin");
        // strcpy(save_vocab_file, "data/text8-vocab.bin");
        cbow = 1;
        layer1_size = 10;
        window = 8;
        negative = 0;
        hs = 1;
        sample = 1e-4;
        num_threads = 1;
        binary = 1;
        iter = 15;
        // return 0;
    }
    else
    {
        int a = 0;
        for(a = 0; a<argc; a++)
        {
            printf("%s\n", argv[a]);
        }

        // 词向量维度
        if ((i = ArgPos((char *)"-size", argc, argv)) > 0)
            // C 库函数 int atoi(const char *str) 把参数 str 所指向的字符串转换为一个整数（类型为 int 型）。
            layer1_size = atoi(argv[i + 1]);
        // 语料库文件
        if ((i = ArgPos((char *)"-train", argc, argv)) > 0)
            strcpy(train_file, argv[i + 1]);
        // 词汇表的保存文件
        if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0)
            strcpy(save_vocab_file, argv[i + 1]);
        // 已有词汇表文件
        if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0)
            strcpy(read_vocab_file, argv[i + 1]);
        // 是否打印信息，大于1表示打印信息
        if ((i = ArgPos((char *)"-debug", argc, argv)) > 0)
            debug_mode = atoi(argv[i + 1]);
        // 训练的词向量的保存结果的形式，是使用二进制还是使用文本进行保存，1表示二进制，0表示文本
        if ((i = ArgPos((char *)"-binary", argc, argv)) > 0)
            binary = atoi(argv[i + 1]);
        // cbow或者skip， 1表示cbow
        if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0)
            cbow = atoi(argv[i + 1]);
        // 如果是cbow，将学习率设置为0.05，如果是skip，一般将学习率设置为0.025
        if (cbow)
            alpha = 0.05;
        // 设置学习率
        if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0)
            alpha = atof(argv[i + 1]);
        // 保存词向量的文件名
        if ((i = ArgPos((char *)"-output", argc, argv)) > 0)
            strcpy(output_file, argv[i + 1]);
        // 窗口大小
        if ((i = ArgPos((char *)"-window", argc, argv)) > 0)
            window = atoi(argv[i + 1]);

        // 下采样率，下采样阀值
        if ((i = ArgPos((char *)"-sample", argc, argv)) > 0)
            sample = atof(argv[i + 1]);
        // 是否使用层次softmax，0表示不使用
        if ((i = ArgPos((char *)"-hs", argc, argv)) > 0)
            hs = atoi(argv[i + 1]);
        // 负采样大小，0表示不用
        if ((i = ArgPos((char *)"-negative", argc, argv)) > 0)
            negative = atoi(argv[i + 1]);
        // 训练采用的线程数
        if ((i = ArgPos((char *)"-threads", argc, argv)) > 0)
            num_threads = atoi(argv[i + 1]);
        // 训练的迭代次数
        if ((i = ArgPos((char *)"-iter", argc, argv)) > 0)
            iter = atoi(argv[i + 1]);
        // 单词的最小频率，小于这个频率的会被舍弃
        if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0)
            min_count = atoi(argv[i + 1]);
        // 聚类中心数
        if ((i = ArgPos((char *)"-classes", argc, argv)) > 0)
            classes = atoi(argv[i + 1]);
    }


    // C 库函数 void *calloc(size_t nitems, size_t size) 分配所需的内存空间，并返回一个指向它的指针。
    // malloc 和 calloc 之间的不同点是，malloc 不会设置内存为零，而 calloc 会设置分配的内存为零。
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

    //初始化expTable，近似逼近sigmoid(x)值，x区间为[-MAX_EXP, MAX_EXP]，分成EXP_TABLE_SIZE份
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++)
    {
        float a = (i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP;
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    TrainModel();
    return 0;
}
