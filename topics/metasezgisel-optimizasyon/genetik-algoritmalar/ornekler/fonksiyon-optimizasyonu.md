---
title: "Genetik Algoritma: Basit Fonksiyon Optimizasyonu (Python)"
description: "Genetik algoritmalar kullanılarak Python'da basit bir x^2 fonksiyonunun nasıl maksimize edileceğine dair adım adım bir örnek."
keywords: "genetik algoritma, python, fonksiyon optimizasyonu, x^2 maksimizasyonu, metasezgisel optimizasyon örneği"
language: "Python"
---

## Basit Bir Fonksiyon Optimizasyonu

**Problem:** Belirli bir aralıkta `f(x) = x^2` fonksiyonunu maksimize eden `x` değerini bulmak. (Bu basit bir örnektir, genetik algoritmalar genellikle daha karmaşık fonksiyonlar için kullanılır.)

**Python Kodu:**

```python
# Gerekli kütüphaneler (örneğin, numpy)
import numpy as np
import random

# Parametreler
population_size = 50
# chromosome_length = 8 # x değerini binary olarak temsil etmek için (Bu örnekte doğrudan float kullanılıyor)
mutation_rate = 0.01
crossover_rate = 0.7
generations = 100
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

# Çaprazlama (Aritmetik Çaprazlama)
def crossover(parent1, parent2, rate):
    if random.random() < rate:
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2
    return parent1, parent2

# Mutasyon
def mutate(individual, rate):
    if random.random() < rate:
        mutation_value = random.uniform(-0.5, 0.5) # Değere küçük bir rastgele sayı ekle/çıkar
        mutated_individual = individual + mutation_value
        return np.clip(mutated_individual, -10, 10) # Aralığın dışına çıkarsa düzelt
    return individual

# Genetik Algoritma Akışı
population = create_population(population_size)

for gen in range(generations):
    fitness_values = [fitness_function(ind) for ind in population]
    
    best_fitness_idx = np.argmax(fitness_values)
    best_individual_this_gen = population[best_fitness_idx] # Değişken adı düzeltildi
    # print(f"Nesil {gen+1}: En İyi Birey = {best_individual_this_gen:.4f}, Uygunluk = {fitness_values[best_fitness_idx]:.4f}") # Çıktı çok uzun olabilir, isteğe bağlı

    parents = selection(population, fitness_values)
    
    next_generation = []
    for i in range(0, population_size, 2):
        parent1 = parents[i]
        # Popülasyon tekse sonuncuyu kendiyle eşle veya listeden rastgele başka bir ebeveyn seç
        parent2 = parents[i+1] if (i+1) < len(parents) else parents[random.randint(0, len(parents)-1)] 
        
        child1, child2 = crossover(parent1, parent2, crossover_rate)
        
        next_generation.append(mutate(child1, mutation_rate))
        if len(next_generation) < population_size:
            next_generation.append(mutate(child2, mutation_rate))
            
    population = next_generation[:population_size] # Popülasyon boyutunu koru

final_fitness_values = [fitness_function(ind) for ind in population]
best_final_fitness_idx = np.argmax(final_fitness_values)
best_solution = population[best_final_fitness_idx]
print(f"\nEn İyi Çözüm: x = {best_solution:.4f}, f(x) = {final_fitness_values[best_final_fitness_idx]:.4f}")
```

**Açıklama:**
Yukarıdaki Python kodu, basit bir genetik algoritma implementasyonunu göstermektedir. `x^2` fonksiyonunu maksimize etmeye çalışır. Gerçek dünya problemlerinde kromozom temsili, uygunluk fonksiyonu ve genetik operatörler (seçilim, çaprazlama, mutasyon) probleme özgü olarak dikkatlice tasarlanmalıdır. 