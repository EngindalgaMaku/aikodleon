import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: 'Python Temel Algoritmalar | Kodleon',
  description: 'Python\'da temel algoritmaları, algoritma analizi, karmaşıklık ve daha fazlasını öğrenin.',
};

const content = `
# Python'da Temel Algoritmalar

Bu bölümde, temel algoritma kavramlarını ve Python'da nasıl uygulandıklarını öğreneceğiz.

## 1. Algoritma Analizi

Algoritmaların performansını değerlendirmek için kullanılan temel kavramlar:

### Big O Notasyonu

\`\`\`python
# O(1) - Sabit Zaman
def get_first_element(arr):
    return arr[0] if arr else None

# O(n) - Doğrusal Zaman
def find_element(arr, target):
    for element in arr:
        if element == target:
            return True
    return False

# O(n²) - Karesel Zaman
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
\`\`\`

## 2. Temel Arama Algoritmaları

### Doğrusal Arama (Linear Search)

\`\`\`python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Kullanım
arr = [64, 34, 25, 12, 22, 11, 90]
result = linear_search(arr, 12)  # 3
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
result = binary_search(arr, 25)  # 3
\`\`\`

## 3. Temel Sıralama Algoritmaları

### Kabarcık Sıralama (Bubble Sort)

\`\`\`python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

# Kullanım
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = bubble_sort(arr)  # [11, 12, 22, 25, 34, 64, 90]
\`\`\`

### Seçmeli Sıralama (Selection Sort)

\`\`\`python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# Kullanım
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = selection_sort(arr)  # [11, 12, 22, 25, 34, 64, 90]
\`\`\`

### Eklemeli Sıralama (Insertion Sort)

\`\`\`python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr

# Kullanım
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = insertion_sort(arr)  # [11, 12, 22, 25, 34, 64, 90]
\`\`\`

## 4. Özyinelemeli Algoritmalar (Recursion)

### Faktöriyel Hesaplama

\`\`\`python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n-1)

# Kullanım
result = factorial(5)  # 120
\`\`\`

### Fibonacci Dizisi

\`\`\`python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Kullanım
result = fibonacci(6)  # 8
\`\`\`

### Üs Alma

\`\`\`python
def power(base, exponent):
    if exponent == 0:
        return 1
    if exponent == 1:
        return base
    return base * power(base, exponent-1)

# Kullanım
result = power(2, 3)  # 8
\`\`\`

## 5. Temel Algoritma Stratejileri

### Brute Force

\`\`\`python
def find_pairs(arr, target):
    pairs = []
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] + arr[j] == target:
                pairs.append((arr[i], arr[j]))
    return pairs

# Kullanım
arr = [1, 2, 3, 4, 5]
pairs = find_pairs(arr, 7)  # [(2, 5), (3, 4)]
\`\`\`

### Divide and Conquer

\`\`\`python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Kullanım
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = merge_sort(arr)  # [11, 12, 22, 25, 34, 64, 90]
\`\`\`

## Alıştırmalar

1. **Arama Algoritmaları**
   - Verilen bir dizide en çok tekrar eden elemanı bulun
   - İkili arama algoritmasını özyinelemeli olarak yazın
   - Eksik sayıyı bulun (1'den n'e kadar olan sayılardan biri eksik)

2. **Sıralama Algoritmaları**
   - Quick Sort algoritmasını implemente edin
   - Verilen bir diziyi hem artan hem azalan sırada sıralayın
   - İki sıralı diziyi tek bir sıralı dizide birleştirin

3. **Özyineleme**
   - Bir sayının palindrom olup olmadığını kontrol edin
   - Bir dizinin toplamını özyinelemeli olarak hesaplayın
   - Bir stringi ters çevirin

## Kaynaklar

- [Python Algoritmaları](https://github.com/TheAlgorithms/Python)
- [GeeksforGeeks - Python Algoritmaları](https://www.geeksforgeeks.org/python-programming-examples/)
- [Visualgo - Algoritma Görselleştirme](https://visualgo.net/)
`;

export default function BasicAlgorithmsPage() {
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
        <Tabs defaultValue="search">
          <TabsList>
            <TabsTrigger value="search">Arama</TabsTrigger>
            <TabsTrigger value="sort">Sıralama</TabsTrigger>
            <TabsTrigger value="recursion">Özyineleme</TabsTrigger>
          </TabsList>
          
          <TabsContent value="search">
            <Card>
              <CardHeader>
                <CardTitle>Arama Algoritmaları</CardTitle>
                <CardDescription>
                  Doğrusal ve İkili Arama örnekleri
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`# Doğrusal Arama
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# İkili Arama
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
          
          <TabsContent value="sort">
            <Card>
              <CardHeader>
                <CardTitle>Sıralama Algoritmaları</CardTitle>
                <CardDescription>
                  Kabarcık ve Seçmeli Sıralama örnekleri
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`# Kabarcık Sıralama
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# Seçmeli Sıralama
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# Test
arr = [64, 34, 25, 12, 22, 11, 90]
print(bubble_sort(arr.copy()))     # [11, 12, 22, 25, 34, 64, 90]
print(selection_sort(arr.copy()))  # [11, 12, 22, 25, 34, 64, 90]`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="recursion">
            <Card>
              <CardHeader>
                <CardTitle>Özyinelemeli Algoritmalar</CardTitle>
                <CardDescription>
                  Faktöriyel ve Fibonacci örnekleri
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`# Faktöriyel
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n-1)

# Fibonacci
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test
print(factorial(5))   # 120
print(fibonacci(6))   # 8`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Navigation */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button asChild variant="outline" className="gap-2">
          <Link href="/topics/python/veri-yapilari-ve-algoritmalar/ileri-veri-yapilari">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: İleri Veri Yapıları
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/veri-yapilari-ve-algoritmalar/siralama-algoritmalari">
            Sonraki Konu: Sıralama Algoritmaları
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