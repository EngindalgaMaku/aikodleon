import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: 'Python İleri Veri Yapıları | Kodleon',
  description: 'Python\'da ileri veri yapılarını, bağlı listeler, ağaçlar, graflar ve daha fazlasını öğrenin.',
};

const content = `
# Python'da İleri Veri Yapıları

Bu bölümde, Python'da daha karmaşık veri yapılarını ve bunların uygulamalarını öğreneceğiz. Bu veri yapıları, daha karmaşık problemleri çözmek için kullanılır.

## 1. Bağlı Listeler (Linked Lists)

Bağlı listeler, her elemanın bir sonraki elemanı işaret ettiği dinamik veri yapılarıdır.

### Tekli Bağlı Liste (Singly Linked List)

\`\`\`python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def display(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

# Kullanım
lst = LinkedList()
lst.append(1)
lst.append(2)
lst.append(3)
lst.display()  # 1 -> 2 -> 3 -> None
\`\`\`

### Çiftli Bağlı Liste (Doubly Linked List)

\`\`\`python
class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        new_node.prev = current
\`\`\`

## 2. Ağaçlar (Trees)

Ağaçlar, hiyerarşik veri yapılarıdır ve birçok uygulamada kullanılır.

### İkili Arama Ağacı (Binary Search Tree)

\`\`\`python
class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, data):
        if not self.root:
            self.root = TreeNode(data)
        else:
            self._insert_recursive(self.root, data)
    
    def _insert_recursive(self, node, data):
        if data < node.data:
            if node.left is None:
                node.left = TreeNode(data)
            else:
                self._insert_recursive(node.left, data)
        else:
            if node.right is None:
                node.right = TreeNode(data)
            else:
                self._insert_recursive(node.right, data)
    
    def inorder_traversal(self):
        def _inorder(node):
            if node:
                _inorder(node.left)
                print(node.data, end=" ")
                _inorder(node.right)
        _inorder(self.root)

# Kullanım
bst = BinarySearchTree()
bst.insert(5)
bst.insert(3)
bst.insert(7)
bst.inorder_traversal()  # 3 5 7
\`\`\`

### AVL Ağacı

\`\`\`python
class AVLNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def get_height(self, node):
        if not node:
            return 0
        return node.height
    
    def get_balance(self, node):
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)
    
    def right_rotate(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = max(self.get_height(y.left), self.get_height(y.right)) + 1
        x.height = max(self.get_height(x.left), self.get_height(x.right)) + 1
        return x
\`\`\`

## 3. Graflar (Graphs)

Graflar, düğümler ve kenarlardan oluşan veri yapılarıdır.

### Komşuluk Matrisi ile Graf

\`\`\`python
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] 
                      for row in range(vertices)]
    
    def add_edge(self, v1, v2):
        self.graph[v1][v2] = 1
        self.graph[v2][v1] = 1
    
    def print_graph(self):
        for i in range(self.V):
            for j in range(self.V):
                print(self.graph[i][j], end=" ")
            print()

# Kullanım
g = Graph(4)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.print_graph()
\`\`\`

### Komşuluk Listesi ile Graf

\`\`\`python
from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v):
        self.graph[u].append(v)
    
    def BFS(self, s):
        visited = [False] * len(self.graph)
        queue = []
        queue.append(s)
        visited[s] = True
        
        while queue:
            s = queue.pop(0)
            print(s, end=" ")
            
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
\`\`\`

## 4. Heap (Yığın)

Heap, öncelikli kuyruk implementasyonu için kullanılan özel bir ağaç yapısıdır.

\`\`\`python
class MinHeap:
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def insert(self, key):
        self.heap.append(key)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        parent = self.parent(i)
        if i > 0 and self.heap[i] < self.heap[parent]:
            self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
            self._heapify_up(parent)
\`\`\`

## 5. Trie (Önek Ağacı)

Trie, string aramaları için optimize edilmiş bir ağaç yapısıdır.

\`\`\`python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
\`\`\`

## Alıştırmalar

1. **Bağlı Liste Problemleri**
   - Bağlı listeyi tersine çevirin
   - Döngü tespit edin
   - İki bağlı listeyi birleştirin

2. **İkili Ağaç Problemleri**
   - Ağacın yüksekliğini bulun
   - Simetrik olup olmadığını kontrol edin
   - En düşük ortak atayı bulun

3. **Graf Problemleri**
   - En kısa yol bulma (Dijkstra)
   - Çevrim tespit etme
   - Graf renklendirme

## Kaynaklar

- [Python Algoritma ve Veri Yapıları](https://github.com/TheAlgorithms/Python)
- [GeeksforGeeks - İleri Veri Yapıları](https://www.geeksforgeeks.org/advanced-data-structures/)
- [Visualgo - Veri Yapıları Görselleştirme](https://visualgo.net/)
`;

export default function AdvancedDataStructuresPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <Button asChild variant="ghost" size="sm" className="gap-1">
          <Link href="/topics/python/veri-yapilari-ve-algoritmalar">
            <ArrowLeft className="h-4 w-4" />
            Veri Yapıları ve Algoritmalara Dön
          </Link>
        </Button>
      </div>
      
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <MarkdownContent content={content} />
      </div>
      
      {/* Interactive Examples */}
      <div className="my-12">
        <h2 className="text-3xl font-bold mb-8">İnteraktif Örnekler</h2>
        <Tabs defaultValue="linkedlist">
          <TabsList>
            <TabsTrigger value="linkedlist">Bağlı Liste</TabsTrigger>
            <TabsTrigger value="bst">İkili Arama Ağacı</TabsTrigger>
            <TabsTrigger value="graph">Graf</TabsTrigger>
          </TabsList>
          
          <TabsContent value="linkedlist">
            <Card>
              <CardHeader>
                <CardTitle>Bağlı Liste İşlemleri</CardTitle>
                <CardDescription>
                  Tekli bağlı liste implementasyonu ve temel işlemler
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

# Bağlı liste oluşturma
head = Node(1)
head.next = Node(2)
head.next.next = Node(3)

# Listeyi yazdırma
current = head
while current:
    print(current.data, end=" -> ")
    current = current.next
print("None")`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="bst">
            <Card>
              <CardHeader>
                <CardTitle>İkili Arama Ağacı İşlemleri</CardTitle>
                <CardDescription>
                  BST oluşturma ve temel işlemler
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

# Ağaç oluşturma
root = TreeNode(5)
root.left = TreeNode(3)
root.right = TreeNode(7)

# Inorder traversal
def inorder(node):
    if node:
        inorder(node.left)
        print(node.data, end=" ")
        inorder(node.right)

inorder(root)  # 3 5 7`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="graph">
            <Card>
              <CardHeader>
                <CardTitle>Graf İşlemleri</CardTitle>
                <CardDescription>
                  Komşuluk matrisi ile graf implementasyonu
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for _ in range(vertices)] 
                      for _ in range(vertices)]
    
    def add_edge(self, v1, v2):
        self.graph[v1][v2] = 1
        self.graph[v2][v1] = 1

# Graf oluşturma
g = Graph(4)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)

# Matrisi yazdırma
for row in g.graph:
    print(row)`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Navigation */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button asChild variant="outline" className="gap-2">
          <Link href="/topics/python/veri-yapilari-ve-algoritmalar/temel-veri-yapilari">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Temel Veri Yapıları
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/veri-yapilari-ve-algoritmalar/temel-algoritmalar">
            Sonraki Konu: Temel Algoritmalar
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Button>
      </div>
      
      <div className="mt-16 text-center text-sm text-muted-foreground">
        <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
      </div>
    </div>
  );
} 