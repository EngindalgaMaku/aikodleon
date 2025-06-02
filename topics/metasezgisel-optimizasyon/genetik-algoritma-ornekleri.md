# Genetik Algoritma Örnekleri (Python)

Bu bölümde, genetik algoritmaların çeşitli problemlere nasıl uygulanabileceğine dair Python ile yazılmış pratik örnekler ve detaylı açıklamalar bulacaksınız. Genetik algoritmalar, optimizasyon ve arama problemlerinde etkili çözümler sunan evrimsel hesaplama teknikleridir.

## Temel Kavramlar

Örneklere geçmeden önce, genetik algoritmalardaki bazı temel kavramları hatırlayalım:

*   **Popülasyon:** Çözüm adaylarından oluşan bir küme.
*   **Kromozom (Birey):** Popülasyondaki her bir çözüm adayı.
*   **Gen:** Bir kromozomu oluşturan temel bilgi birimi.
*   **Uygunluk Fonksiyonu (Fitness Function):** Bir çözüm adayının ne kadar iyi olduğunu değerlendiren fonksiyon.
*   **Seçilim (Selection):** Daha iyi çözüm adaylarının bir sonraki nesle aktarılma olasılığının artırıldığı süreç.
*   **Çaprazlama (Crossover):** İki ebeveyn kromozomdan yeni çocuk kromozomlar üretme işlemi.
*   **Mutasyon (Mutation):** Kromozomlardaki genlerde rastgele değişiklikler yaparak çeşitliliği artırma işlemi.

## Örnek 1: Basit Bir Fonksiyon Optimizasyonu

**Problem:** Belirli bir aralıkta `f(x) = x^2` fonksiyonunu maksimize eden `x` değerini bulmak. (Bu basit bir örnektir, genetik algoritmalar genellikle daha karmaşık fonksiyonlar için kullanılır.)

**Python Kodu:**

```python
# Gerekli kütüphaneler (örneğin, numpy)
import numpy as np
import random

# Parametreler
population_size = 50
chromosome_length = 8 # x değerini binary olarak temsil etmek için
mutation_rate = 0.01
crossover_rate = 0.7
generations = 100
# x değeri için aralık (örneğin 0-15 arası tamsayılar için 4 bit yeterli olabilir, 
# daha geniş aralıklar ve hassasiyet için bit sayısı artırılmalı)
# Bu örnekte -10 ile 10 arasında bir değer arayalım ve bunu temsil edelim.
# Kromozom, doğrudan bir float değeri temsil etsin bu basit örnekte.
# Aralık [-10, 10]

# Uygunluk Fonksiyonu
def fitness_function(chromosome_val):
    # x^2 fonksiyonunu maksimize etmeye çalışıyoruz
    return chromosome_val**2

# Popülasyon Oluşturma
def create_individual():
    # [-10, 10] aralığında rastgele bir float değer
    return random.uniform(-10, 10)

def create_population(size):
    return [create_individual() for _ in range(size)]

# Seçilim (Turnuva Seçimi)
def selection(population, fitness_values):
    tournament_size = 3
    selected_parents = []
    for _ in range(len(population)):
        tournament_candidates_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_candidates_indices]
        winner_index_in_tournament = np.argmax(tournament_fitness)
        selected_parents.append(population[tournament_candidates_indices[winner_index_in_tournament]])
    return selected_parents

# Çaprazlama (Tek Noktalı veya Aritmetik)
def crossover(parent1, parent2, rate):
    if random.random() < rate:
        # Aritmetik çaprazlama
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2
    return parent1, parent2

# Mutasyon
def mutate(individual, rate):
    if random.random() < rate:
        # Değere küçük bir rastgele sayı ekle/çıkar
        mutation_value = random.uniform(-0.5, 0.5)
        mutated_individual = individual + mutation_value
        # Aralığın dışına çıkarsa düzelt (opsiyonel, probleme göre değişir)
        return np.clip(mutated_individual, -10, 10)
    return individual

# Genetik Algoritma Akışı
population = create_population(population_size)

for gen in range(generations):
    fitness_values = [fitness_function(ind) for ind in population]
    
    # En iyi bireyi bul ve yazdır
    best_fitness_idx = np.argmax(fitness_values)
    best_individual = population[best_fitness_idx]
    print(f"Nesil {gen+1}: En İyi Birey = {best_individual:.4f}, Uygunluk = {fitness_values[best_fitness_idx]:.4f}")

    parents = selection(population, fitness_values)
    
    next_generation = []
    for i in range(0, population_size, 2):
        parent1 = parents[i]
        parent2 = parents[i+1 if i+1 < population_size else i] # Popülasyon tekse sonuncuyu kendiyle eşle
        
        child1, child2 = crossover(parent1, parent2, crossover_rate)
        
        next_generation.append(mutate(child1, mutation_rate))
        if len(next_generation) < population_size:
            next_generation.append(mutate(child2, mutation_rate))
            
    population = next_generation[:population_size]

final_fitness_values = [fitness_function(ind) for ind in population]
best_final_fitness_idx = np.argmax(final_fitness_values)
best_solution = population[best_final_fitness_idx]
print(f"\nEn İyi Çözüm: x = {best_solution:.4f}, f(x) = {final_fitness_values[best_final_fitness_idx]:.4f}")

```
**Açıklama:**
Yukarıdaki Python kodu, basit bir genetik algoritma implementasyonunu göstermektedir. `x^2` fonksiyonunu maksimize etmeye çalışır. Gerçek dünya problemlerinde kromozom temsili, uygunluk fonksiyonu ve genetik operatörler (seçilim, çaprazlama, mutasyon) probleme özgü olarak dikkatlice tasarlanmalıdır.

## Örnek 2: Gezgin Satıcı Problemi (TSP) için Basit Bir Yaklaşım

**Problem:** Bir grup şehir ve aralarındaki mesafeler verildiğinde, her şehri tam olarak bir kez ziyaret edip başlangıç şehrine dönen en kısa turu bulmak.

**(Python kodu ve açıklaması buraya eklenecek...)**

## Örnek 3: Sırt Çantası Problemi (Knapsack Problem)

**Problem:** Bir sırt çantasının belirli bir ağırlık kapasitesi vardır. Bir dizi eşya ve her bir eşyanın ağırlığı ile değeri verilmiştir. Sırt çantasına, toplam ağırlık kapasitesini aşmayacak şekilde, toplam değeri maksimize eden eşyaları yerleştirmek.

**(Python kodu ve açıklaması buraya eklenecek...)**

---

Bu sayfa, genetik algoritmaların pratik uygulamalarına bir giriş niteliğindedir. Daha karmaşık problemler ve gelişmiş teknikler için ileri düzey kaynaklara başvurmanız önerilir. 