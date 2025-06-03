import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: 'Python Dinamik Programlama | Kodleon',
  description: 'Python\'da dinamik programlama, memoization, tabulasyon ve optimizasyon tekniklerini öğrenin.',
};

const content = `
# Python'da Dinamik Programlama

Bu bölümde, karmaşık problemleri alt problemlere bölerek çözen dinamik programlama yaklaşımını öğreneceğiz.

## 1. Dinamik Programlama Temelleri

### Memoization (Üstten Aşağı Yaklaşım)

\`\`\`python
# Fibonacci - Naive Recursive
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n-1) + fib_recursive(n-2)

# Fibonacci - Memoization
def fib_memo(n, memo=None):
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# Kullanım
n = 10
print(fib_memo(n))  # 55
\`\`\`

### Tabulation (Alttan Yukarı Yaklaşım)

\`\`\`python
def fib_tabulation(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# Kullanım
n = 10
print(fib_tabulation(n))  # 55
\`\`\`

## 2. Klasik Dinamik Programlama Problemleri

### En Uzun Artan Alt Dizi (LIS)

\`\`\`python
def longest_increasing_subsequence(arr):
    if not arr:
        return 0
    
    n = len(arr)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(0, i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# Kullanım
arr = [10, 22, 9, 33, 21, 50, 41, 60, 80]
print(longest_increasing_subsequence(arr))  # 6
\`\`\`

### Sırt Çantası Problemi (Knapsack)

\`\`\`python
def knapsack(values, weights, capacity):
    n = len(values)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], 
                              dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]

# Kullanım
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50
print(knapsack(values, weights, capacity))  # 220
\`\`\`

### En Uzun Ortak Alt Dizi (LCS)

\`\`\`python
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# Kullanım
text1 = "ABCDGH"
text2 = "AEDFHR"
print(longest_common_subsequence(text1, text2))  # 3
\`\`\`

## 3. Matris Zinciri Çarpımı

\`\`\`python
def matrix_chain_multiplication(dimensions):
    n = len(dimensions) - 1
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = (dp[i][k] + dp[k+1][j] + 
                       dimensions[i] * dimensions[k+1] * dimensions[j+1])
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[0][n-1]

# Kullanım
dimensions = [10, 30, 5, 60]
print(matrix_chain_multiplication(dimensions))  # 4500
\`\`\`

## 4. Yol Bulma Problemleri

### En Kısa Yol Toplamı

\`\`\`python
def min_path_sum(grid):
    if not grid:
        return 0
    
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    
    # İlk satırı doldur
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    # İlk sütunu doldur
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    
    # Geri kalan hücreleri doldur
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    
    return dp[m-1][n-1]

# Kullanım
grid = [
    [1, 3, 1],
    [1, 5, 1],
    [4, 2, 1]
]
print(min_path_sum(grid))  # 7
\`\`\`

### Benzersiz Yollar

\`\`\`python
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]

# Kullanım
m, n = 3, 7
print(unique_paths(m, n))  # 28
\`\`\`

## 5. Optimizasyon Teknikleri

### Uzay Optimizasyonu

\`\`\`python
def fibonacci_optimized(n):
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1

# Kullanım
n = 10
print(fibonacci_optimized(n))  # 55
\`\`\`

### Durum Geçişi Optimizasyonu

\`\`\`python
def coin_change_optimized(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Kullanım
coins = [1, 2, 5]
amount = 11
print(coin_change_optimized(coins, amount))  # 3
\`\`\`

## 6. Pratik Uygulamalar

### Metin Düzenleme Mesafesi

\`\`\`python
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j],      # silme
                             dp[i][j-1],        # ekleme
                             dp[i-1][j-1]) + 1  # değiştirme
    
    return dp[m][n]

# Kullanım
word1 = "horse"
word2 = "ros"
print(edit_distance(word1, word2))  # 3
\`\`\`

## Alıştırmalar

1. **Temel Problemler**
   - Merdiven tırmanma problemi
   - Maksimum alt dizi toplamı
   - Palindrom alt dizileri

2. **Orta Seviye**
   - Bölünebilir alt küme problemi
   - Düzenli ifade eşleştirme
   - En uzun palindromik alt dizi

3. **İleri Seviye**
   - Maksimum kare matrisi
   - Zar atma olasılıkları
   - Stok alım-satım problemi

## Kaynaklar

- [Dynamic Programming Patterns](https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns)
- [GeeksforGeeks - Dynamic Programming](https://www.geeksforgeeks.org/dynamic-programming/)
- [Algorithms for Competitive Programming](https://cp-algorithms.com/dynamic_programming/)
`;

export default function DynamicProgrammingPage() {
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
            <TabsTrigger value="classic">Klasik</TabsTrigger>
            <TabsTrigger value="optimization">Optimizasyon</TabsTrigger>
          </TabsList>
          
          <TabsContent value="basic">
            <Card>
              <CardHeader>
                <CardTitle>Fibonacci Hesaplama</CardTitle>
                <CardDescription>
                  Memoization ve Tabulation yaklaşımları
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`# Memoization
def fib_memo(n, memo=None):
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# Tabulation
def fib_tab(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# Test
n = 10
print(f"Memoization: {fib_memo(n)}")  # 55
print(f"Tabulation: {fib_tab(n)}")    # 55`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="classic">
            <Card>
              <CardHeader>
                <CardTitle>Klasik DP Problemleri</CardTitle>
                <CardDescription>
                  LIS ve Knapsack örnekleri
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`# En Uzun Artan Alt Dizi
def lis(arr):
    n = len(arr)
    dp = [1] * n
    for i in range(1, n):
        for j in range(0, i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# Sırt Çantası
def knapsack(values, weights, capacity):
    n = len(values)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], 
                              dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]

# Test
arr = [10, 22, 9, 33, 21, 50, 41, 60]
print(f"LIS: {lis(arr)}")  # 5

values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50
print(f"Knapsack: {knapsack(values, weights, capacity)}")  # 220`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="optimization">
            <Card>
              <CardHeader>
                <CardTitle>Optimizasyon Teknikleri</CardTitle>
                <CardDescription>
                  Uzay ve durum geçişi optimizasyonları
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`# Uzay Optimizasyonlu Fibonacci
def fib_optimized(n):
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    return prev1

# Para Üstü Problemi
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

# Test
print(f"Fibonacci(10): {fib_optimized(10)}")  # 55

coins = [1, 2, 5]
amount = 11
print(f"Coin Change: {coin_change(coins, amount)}")  # 3`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Navigation */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button asChild variant="outline" className="gap-2">
          <Link href="/topics/python/veri-yapilari-ve-algoritmalar/arama-algoritmalari">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Arama Algoritmaları
          </Link>
        </Button>
        
        <Button variant="outline" disabled className="gap-2">
          Sonraki Konu
          <ArrowRight className="h-4 w-4" />
        </Button>
      </div>
      
      <div className="mt-16 text-center text-sm text-muted-foreground">
        <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
      </div>
    </div>
  );
} 