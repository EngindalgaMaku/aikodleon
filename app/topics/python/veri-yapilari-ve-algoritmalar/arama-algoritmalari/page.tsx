import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: 'Python Arama Algoritmaları | Kodleon',
  description: 'Python\'da arama algoritmalarını, graf arama, örüntü eşleştirme ve daha fazlasını öğrenin.',
};

const content = `
# Python'da Arama Algoritmaları

Bu bölümde, Python'da farklı arama algoritmalarını ve bunların uygulamalarını öğreneceğiz.

## 1. Temel Arama Algoritmaları

### Doğrusal Arama (Linear Search)

\`\`\`python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Kullanım
arr = [64, 34, 25, 12, 22, 11, 90]
index = linear_search(arr, 22)  # 4
\`\`\`

### İkili Arama (Binary Search)

\`\`\`python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Kullanım
arr = [11, 12, 22, 25, 34, 64, 90]
index = binary_search(arr, 25)  # 3
\`\`\`

### Sıçramalı Arama (Jump Search)

\`\`\`python
import math

def jump_search(arr, target):
    n = len(arr)
    step = int(math.sqrt(n))
    
    # Finding the block where element is present
    prev = 0
    while arr[min(step, n)-1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    
    # Linear search in block beginning with prev
    while arr[prev] < target:
        prev += 1
        if prev == min(step, n):
            return -1
    
    if arr[prev] == target:
        return prev
    
    return -1

# Kullanım
arr = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
index = jump_search(arr, 55)  # 10
\`\`\`

## 2. Graf Arama Algoritmaları

### Derinlik Öncelikli Arama (DFS)

\`\`\`python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start, end=' ')
    
    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited

# Kullanım
graph = {
    'A': {'B', 'C'},
    'B': {'A', 'D', 'E'},
    'C': {'A', 'F'},
    'D': {'B'},
    'E': {'B', 'F'},
    'F': {'C', 'E'}
}

dfs(graph, 'A')  # A B D E F C
\`\`\`

### Genişlik Öncelikli Arama (BFS)

\`\`\`python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')
        
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

# Kullanım
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

bfs(graph, 'A')  # A B C D E F
\`\`\`

## 3. Örüntü Eşleştirme Algoritmaları

### Naive String Search

\`\`\`python
def naive_search(text, pattern):
    n = len(text)
    m = len(pattern)
    positions = []
    
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            positions.append(i)
    
    return positions

# Kullanım
text = "AABAACAADAABAAABAA"
pattern = "AABA"
positions = naive_search(text, pattern)  # [0, 9, 13]
\`\`\`

### KMP (Knuth-Morris-Pratt) Algoritması

\`\`\`python
def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length-1]
            else:
                lps[i] = 0
                i += 1
    return lps

def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    positions = []
    
    lps = compute_lps(pattern)
    i = j = 0
    
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == m:
            positions.append(i-j)
            j = lps[j-1]
        
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
    
    return positions

# Kullanım
text = "AABAACAADAABAAABAA"
pattern = "AABA"
positions = kmp_search(text, pattern)  # [0, 9, 13]
\`\`\`

## 4. Özel Arama Algoritmaları

### Interpolation Search

\`\`\`python
def interpolation_search(arr, target):
    low = 0
    high = len(arr) - 1
    
    while low <= high and target >= arr[low] and target <= arr[high]:
        if low == high:
            if arr[low] == target:
                return low
            return -1
        
        pos = low + int(((float(high - low) / 
            (arr[high] - arr[low])) * (target - arr[low])))
        
        if arr[pos] == target:
            return pos
        
        if arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1
    
    return -1

# Kullanım
arr = [10, 12, 13, 16, 18, 19, 20, 21, 22, 23, 24, 33, 35, 42, 47]
index = interpolation_search(arr, 18)  # 4
\`\`\`

### Exponential Search

\`\`\`python
def exponential_search(arr, target):
    if arr[0] == target:
        return 0
    
    n = len(arr)
    i = 1
    while i < n and arr[i] <= target:
        i = i * 2
    
    return binary_search(arr, target, i//2, min(i, n-1))

def binary_search(arr, target, left, right):
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Kullanım
arr = [2, 3, 4, 10, 40]
index = exponential_search(arr, 10)  # 3
\`\`\`

## 5. Karmaşıklık Analizi

| Algoritma | En İyi Durum | Ortalama Durum | En Kötü Durum | Bellek |
|-----------|-------------|----------------|---------------|---------|
| Linear Search | O(1) | O(n) | O(n) | O(1) |
| Binary Search | O(1) | O(log n) | O(log n) | O(1) |
| Jump Search | O(1) | O(√n) | O(√n) | O(1) |
| DFS | O(V + E) | O(V + E) | O(V + E) | O(V) |
| BFS | O(V + E) | O(V + E) | O(V + E) | O(V) |
| Naive String | O(n) | O(mn) | O(mn) | O(1) |
| KMP | O(n) | O(n) | O(n) | O(m) |
| Interpolation | O(1) | O(log log n) | O(n) | O(1) |
| Exponential | O(1) | O(log n) | O(log n) | O(1) |

## 6. Kullanım Senaryoları

1. **Doğrusal Arama**
   - Küçük veri setleri
   - Sırasız veriler
   - Tek seferlik aramalar

2. **İkili Arama**
   - Sıralı büyük veri setleri
   - Tekrarlı aramalar
   - Log n karmaşıklık gerektiğinde

3. **Sıçramalı Arama**
   - Sıralı büyük veri setleri
   - Bellek kısıtlaması olan durumlar
   - Kare kök karmaşıklık yeterli olduğunda

4. **DFS**
   - Yol bulma problemleri
   - Bağlantı kontrolü
   - Topolojik sıralama

5. **BFS**
   - En kısa yol problemleri
   - Sosyal ağ analizi
   - Web crawler

6. **KMP**
   - Metin içinde örüntü arama
   - DNA dizisi eşleştirme
   - Log dosyası analizi

## Alıştırmalar

1. **Temel Arama**
   - İkili aramayı özyinelemeli olarak implemente edin
   - Tekrar eden elemanları bulan bir arama yazın
   - Rotasyonlu dizide arama yapın

2. **Graf Arama**
   - DFS ile çevrim tespiti yapın
   - BFS ile en kısa yol bulun
   - İki düğüm arası bağlantı kontrolü yapın

3. **Örüntü Eşleştirme**
   - Birden çok örüntüyü aynı anda arayın
   - Bulanık eşleştirme yapın
   - Regex motoru implemente edin

## Kaynaklar

- [Python Search Algorithms](https://github.com/TheAlgorithms/Python/tree/master/searches)
- [GeeksforGeeks - Searching Algorithms](https://www.geeksforgeeks.org/searching-algorithms/)
- [Visualgo - Search Algorithms](https://visualgo.net/en/search)
`;

export default function SearchAlgorithmsPage() {
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
        <Tabs defaultValue="basic">
          <TabsList>
            <TabsTrigger value="basic">Temel</TabsTrigger>
            <TabsTrigger value="graph">Graf</TabsTrigger>
            <TabsTrigger value="pattern">Örüntü</TabsTrigger>
          </TabsList>
          
          <TabsContent value="basic">
            <Card>
              <CardHeader>
                <CardTitle>Temel Arama Algoritmaları</CardTitle>
                <CardDescription>
                  Doğrusal ve İkili Arama örnekleri
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Test
arr = [11, 12, 22, 25, 34, 64, 90]
print(linear_search(arr, 25))  # 3
print(binary_search(arr, 25))  # 3`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="graph">
            <Card>
              <CardHeader>
                <CardTitle>Graf Arama Algoritmaları</CardTitle>
                <CardDescription>
                  DFS ve BFS örnekleri
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`from collections import deque

def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=' ')
    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

# Test
graph = {
    'A': {'B', 'C'},
    'B': {'A', 'D'},
    'C': {'A', 'D'},
    'D': {'B', 'C'}
}
print("DFS:", end=' ')
dfs(graph, 'A')  # A B D C
print("\\nBFS:", end=' ')
bfs(graph, 'A')  # A B C D`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="pattern">
            <Card>
              <CardHeader>
                <CardTitle>Örüntü Eşleştirme</CardTitle>
                <CardDescription>
                  Naive ve KMP algoritmaları
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`def naive_search(text, pattern):
    n, m = len(text), len(pattern)
    positions = []
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            positions.append(i)
    return positions

def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length-1]
            else:
                lps[i] = 0
                i += 1
    return lps

# Test
text = "AABAACAADAABAAABAA"
pattern = "AABA"
print("Naive:", naive_search(text, pattern))  # [0, 9, 13]`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Navigation */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button asChild variant="outline" className="gap-2">
          <Link href="/topics/python/veri-yapilari-ve-algoritmalar/siralama-algoritmalari">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Sıralama Algoritmaları
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/veri-yapilari-ve-algoritmalar/dinamik-programlama">
            Sonraki Konu: Dinamik Programlama
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