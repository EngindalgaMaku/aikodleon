import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: 'Python Sıralama Algoritmaları | Kodleon',
  description: 'Python\'da sıralama algoritmalarını, karmaşıklık analizlerini ve performans karşılaştırmalarını öğrenin.',
};

const content = `
# Python'da Sıralama Algoritmaları

Bu bölümde, Python'da farklı sıralama algoritmalarını, karmaşıklık analizlerini ve kullanım senaryolarını öğreneceğiz.

## 1. Karşılaştırma Tabanlı Sıralama Algoritmaları

### Quick Sort (Hızlı Sıralama)

\`\`\`python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# Kullanım
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = quick_sort(arr)  # [11, 12, 22, 25, 34, 64, 90]
\`\`\`

### Merge Sort (Birleştirme Sıralaması)

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

### Heap Sort (Yığın Sıralaması)

\`\`\`python
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    
    # Build max heap
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements from heap
    for i in range(n-1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    
    return arr

# Kullanım
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = heap_sort(arr)  # [11, 12, 22, 25, 34, 64, 90]
\`\`\`

## 2. Doğrusal Zamanlı Sıralama Algoritmaları

### Counting Sort (Sayma Sıralaması)

\`\`\`python
def counting_sort(arr):
    if not arr:
        return arr
    
    # Find range of array elements
    max_element = max(arr)
    min_element = min(arr)
    range_of_elements = max_element - min_element + 1
    
    # Create a count array to store count of each element
    count = [0] * range_of_elements
    output = [0] * len(arr)
    
    # Store count of each element
    for i in range(len(arr)):
        count[arr[i] - min_element] += 1
    
    # Change count[i] so that count[i] now contains actual
    # position of this element in output array
    for i in range(1, len(count)):
        count[i] += count[i-1]
    
    # Build the output array
    for i in range(len(arr)-1, -1, -1):
        output[count[arr[i] - min_element] - 1] = arr[i]
        count[arr[i] - min_element] -= 1
    
    # Copy the output array to arr
    for i in range(len(arr)):
        arr[i] = output[i]
    
    return arr

# Kullanım
arr = [4, 2, 2, 8, 3, 3, 1]
sorted_arr = counting_sort(arr)  # [1, 2, 2, 3, 3, 4, 8]
\`\`\`

### Radix Sort (Taban Sıralaması)

\`\`\`python
def counting_sort_for_radix(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1
    
    for i in range(1, 10):
        count[i] += count[i-1]
    
    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
    
    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr):
    max_num = max(arr)
    exp = 1
    while max_num // exp > 0:
        counting_sort_for_radix(arr, exp)
        exp *= 10
    return arr

# Kullanım
arr = [170, 45, 75, 90, 802, 24, 2, 66]
sorted_arr = radix_sort(arr)  # [2, 24, 45, 66, 75, 90, 170, 802]
\`\`\`

### Bucket Sort (Kova Sıralaması)

\`\`\`python
def bucket_sort(arr):
    if not arr:
        return arr
    
    # Find minimum and maximum values
    max_val, min_val = max(arr), min(arr)
    
    # Number of buckets, using the difference between max and min
    bucket_range = (max_val - min_val) / len(arr)
    buckets = [[] for _ in range(len(arr) + 1)]
    
    # Put array elements in different buckets
    for i in arr:
        diff = (i - min_val) / bucket_range
        bucket_index = int(diff)
        if bucket_index != len(arr):
            buckets[bucket_index].append(i)
        else:
            buckets[len(arr)-1].append(i)
    
    # Sort individual buckets
    for i in range(len(buckets)):
        buckets[i].sort()
    
    # Concatenate all buckets into arr
    k = 0
    for i in range(len(buckets)):
        for j in range(len(buckets[i])):
            arr[k] = buckets[i][j]
            k += 1
    
    return arr

# Kullanım
arr = [0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434]
sorted_arr = bucket_sort(arr)  # [0.1234, 0.3434, 0.565, 0.656, 0.665, 0.897]
\`\`\`

## 3. Karmaşıklık Analizi

| Algoritma | En İyi Durum | Ortalama Durum | En Kötü Durum | Bellek |
|-----------|-------------|----------------|---------------|---------|
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) |
| Counting Sort | O(n + k) | O(n + k) | O(n + k) | O(k) |
| Radix Sort | O(d(n + k)) | O(d(n + k)) | O(d(n + k)) | O(n + k) |
| Bucket Sort | O(n + k) | O(n + k) | O(n²) | O(n) |

## 4. Kullanım Senaryoları

1. **Quick Sort**
   - Genel amaçlı sıralama
   - Bellek kısıtlaması olan durumlar
   - Ortalama durumda en iyi performans

2. **Merge Sort**
   - Kararlı sıralama gerektiğinde
   - Büyük veri setleri
   - Paralel programlama

3. **Heap Sort**
   - Bellek kısıtlaması olan durumlar
   - Garantili O(n log n) performans
   - Priority Queue implementasyonları

4. **Counting Sort**
   - Sınırlı aralıkta tamsayılar
   - Çok sayıda tekrar eden eleman
   - Doğrusal zaman karmaşıklığı gerektiğinde

5. **Radix Sort**
   - Sabit uzunlukta tamsayılar
   - Stringler
   - Doğrusal zaman karmaşıklığı gerektiğinde

6. **Bucket Sort**
   - Düzgün dağılmış veriler
   - Floating point sayılar
   - Paralel programlama

## Alıştırmalar

1. **Algoritma İyileştirmeleri**
   - Quick Sort için farklı pivot seçim stratejileri deneyin
   - Merge Sort'u yerinde (in-place) yapın
   - Hybrid sıralama algoritması geliştirin

2. **Özel Durumlar**
   - Tekrar eden elemanları olan dizileri sıralayın
   - Çok büyük dosyaları sıralayın (external sorting)
   - Neredeyse sıralı dizileri optimize edin

3. **Karşılaştırmalar**
   - Farklı boyutlarda diziler için performans testi yapın
   - Bellek kullanımını ölçün
   - Kararlılık özelliğini test edin

## Kaynaklar

- [Python Sorting Algorithms](https://github.com/TheAlgorithms/Python/tree/master/sorts)
- [Sorting Algorithms Visualizations](https://www.sortvisualizer.com/)
- [GeeksforGeeks - Sorting Algorithms](https://www.geeksforgeeks.org/sorting-algorithms/)
`;

export default function SortingAlgorithmsPage() {
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
        <Tabs defaultValue="comparison">
          <TabsList>
            <TabsTrigger value="comparison">Karşılaştırmalı</TabsTrigger>
            <TabsTrigger value="linear">Doğrusal</TabsTrigger>
            <TabsTrigger value="special">Özel</TabsTrigger>
          </TabsList>
          
          <TabsContent value="comparison">
            <Card>
              <CardHeader>
                <CardTitle>Karşılaştırma Tabanlı Sıralama</CardTitle>
                <CardDescription>
                  Quick Sort ve Merge Sort örnekleri
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`# Quick Sort
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# Merge Sort
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

# Test
arr = [64, 34, 25, 12, 22, 11, 90]
print(quick_sort(arr.copy()))   # [11, 12, 22, 25, 34, 64, 90]
print(merge_sort(arr.copy()))   # [11, 12, 22, 25, 34, 64, 90]`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="linear">
            <Card>
              <CardHeader>
                <CardTitle>Doğrusal Zamanlı Sıralama</CardTitle>
                <CardDescription>
                  Counting Sort ve Radix Sort örnekleri
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`# Counting Sort
def counting_sort(arr):
    max_val = max(arr)
    count = [0] * (max_val + 1)
    for num in arr:
        count[num] += 1
    i = 0
    for j in range(len(count)):
        while count[j] > 0:
            arr[i] = j
            i += 1
            count[j] -= 1
    return arr

# Test
arr = [4, 2, 2, 8, 3, 3, 1]
print(counting_sort(arr.copy()))  # [1, 2, 2, 3, 3, 4, 8]`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="special">
            <Card>
              <CardHeader>
                <CardTitle>Özel Sıralama Durumları</CardTitle>
                <CardDescription>
                  Bucket Sort örneği
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`# Bucket Sort
def bucket_sort(arr):
    buckets = [[] for _ in range(len(arr))]
    
    # Dağıtma
    for num in arr:
        index = int(num * len(arr))
        buckets[index].append(num)
    
    # Her kovayı sıralama
    for bucket in buckets:
        bucket.sort()
    
    # Birleştirme
    result = []
    for bucket in buckets:
        result.extend(bucket)
    return result

# Test
arr = [0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434]
print(bucket_sort(arr))  # [0.1234, 0.3434, 0.565, 0.656, 0.665, 0.897]`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Navigation */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button asChild variant="outline" className="gap-2">
          <Link href="/topics/python/veri-yapilari-ve-algoritmalar/temel-algoritmalar">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Temel Algoritmalar
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/veri-yapilari-ve-algoritmalar/arama-algoritmalari">
            Sonraki Konu: Arama Algoritmaları
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