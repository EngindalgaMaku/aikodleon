// This file is UTF-8 encoded
import { Metadata } from "next";
import Link from "next/link";
import Image from "next/image";
import { ArrowLeft, ArrowRight, Download, Github, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: "Reinforcement Learning ile Oyun AI | Kod Ornekleri | Kodleon",
  description: "OpenAI Gym kullanarak basit bir oyun icin reinforcement learning ajani gelistirme ornegi.",
  openGraph: {
    title: "Reinforcement Learning ile Oyun AI | Kodleon",
    description: "OpenAI Gym kullanarak basit bir oyun icin reinforcement learning ajani gelistirme ornegi.",
    images: [{ url: "/images/code-examples/reinforcement-learning.jpg" }],
  },
};

export default function ReinforcementLearningPage() {
  return (
    <div className="container max-w-6xl mx-auto py-12 px-4">
      <div className="mb-8">
        <Button asChild variant="outline" size="sm" className="gap-1">
          <Link href="/kod-ornekleri">
            <ArrowLeft className="h-4 w-4" />
            Tum Kod Ornekleri
          </Link>
        </Button>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Sol Taraf - Aciklama */}
        <div className="lg:col-span-1">
          <div className="sticky top-20">
            <h1 className="text-3xl font-bold mb-4">Reinforcement Learning ile Oyun AI</h1>
            
            <div className="flex items-center gap-2 mb-4">
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                Reinforcement Learning
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
                Ileri
              </span>
            </div>
            
            <p className="text-muted-foreground mb-6">
              Bu ornekte, OpenAI Gym kutuphanesini kullanarak basit bir oyun ortaminda reinforcement learning ajani gelistirmeyi ogreneceksiniz. 
              Q-learning algoritmasi ile ajanin ortami kesfederek optimal davranislari ogrenmesini saglayacaksiniz.
            </p>
            
            <div className="space-y-4 mb-6">
              <div>
                <h3 className="font-medium">Gereksinimler:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Python 3.6+</li>
                  <li>OpenAI Gym</li>
                  <li>NumPy</li>
                  <li>Matplotlib</li>
                </ul>
              </div>
              
              <div>
                <h3 className="font-medium">Ogrenilecek Kavramlar:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Reinforcement learning temelleri</li>
                  <li>Q-learning algoritmasi</li>
                  <li>Kesif ve somuru dengesi (Exploration vs. Exploitation)</li>
                  <li>Odul fonksiyonlari</li>
                  <li>Politika optimizasyonu</li>
                </ul>
              </div>
            </div>
            
            <div className="flex flex-col gap-2">
              <Button asChild variant="default" className="gap-2">
                <a href="/notebooks/reinforcement-learning.ipynb" download>
                  <Download className="h-4 w-4" />
                  Jupyter Notebook Indir
                </a>
              </Button>
              <Button asChild variant="outline" className="gap-2">
                <a href="https://github.com/kodleon/ai-examples/blob/main/reinforcement-learning/q-learning-game.ipynb" target="_blank" rel="noopener noreferrer">
                  <Github className="h-4 w-4" />
                  GitHub&apos;da Goruntule
                </a>
              </Button>
            </div>
          </div>
        </div>
        
        {/* Sag Taraf - Kod */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-850 rounded-xl shadow-md overflow-hidden">
            <Tabs defaultValue="code" className="w-full">
              <div className="border-b">
                <TabsList className="p-0 bg-transparent">
                  <TabsTrigger value="code" className="rounded-none py-3 px-6 data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">
                    Kod
                  </TabsTrigger>
                  <TabsTrigger value="explanation" className="rounded-none py-3 px-6 data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">
                    Aciklama
                  </TabsTrigger>
                  <TabsTrigger value="output" className="rounded-none py-3 px-6 data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">
                    Cikti
                  </TabsTrigger>
                </TabsList>
              </div>
              
              <TabsContent value="code" className="p-0 m-0">
                <div className="relative">
                  <Button variant="ghost" size="sm" className="absolute right-2 top-2 gap-1">
                    <Copy className="h-4 w-4" />
                    Kopyala
                  </Button>
                  <pre className="p-6 pt-12 overflow-x-auto text-sm">
                    <code>{`import numpy as np
import gym
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

# OpenAI Gym ortamini olustur - FrozenLake-v1
# Bu oyun, ajanin buz uzerinde kaymadan hedef noktaya ulasmaya calistigi bir oyundur
env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human')

# Q-tablosunu sifirla
# Q-tablosu, her durum-eylem cifti icin bir deger tutar
action_size = env.action_space.n
state_size = env.observation_space.n
q_table = np.zeros((state_size, action_size))

# Hiperparametreler
total_episodes = 15000      # Toplam egitim bolumu sayisi
learning_rate = 0.8         # Ogrenme orani
max_steps = 99              # Bir bolumdeki maksimum adim sayisi
gamma = 0.95                # Gelecekteki odullerin indirim faktoru

# Kesif parametreleri
epsilon = 1.0               # Kesif orani
max_epsilon = 1.0           # Maksimum kesif olasiligi
min_epsilon = 0.01          # Minimum kesif olasiligi
decay_rate = 0.001          # Ussel kesif orani azalma hizi

# Q-learning algoritmasi
def q_learning(env, q_table, learning_rate, gamma, epsilon, max_epsilon, min_epsilon, decay_rate, total_episodes, max_steps):
    # Performans takibi icin listeler
    rewards = []
    epsilons = []
    
    # Her bolum icin
    for episode in range(total_episodes):
        # Ortami sifirla
        state = env.reset()[0]
        step = 0
        done = False
        total_rewards = 0
        
        # Her adim icin
        for step in range(max_steps):
            # Epsilon-greedy stratejisi ile eylem sec
            exp_exp_tradeoff = np.random.uniform(0, 1)
            
            # Kesif
            if exp_exp_tradeoff < epsilon:
                action = env.action_space.sample()
            # Somuru
            else:
                action = np.argmax(q_table[state, :])
                
            # Eylemi gerceklestir ve yeni durumu, odulu ve bitti bilgisini al
            new_state, reward, done, truncated, info = env.step(action)
            
            # Q-tablosunu guncelle (Bellman denklemi)
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
            
            # Durumu guncelle
            state = new_state
            
            # Odulu topla
            total_rewards += reward
            
            # Eger bolum bittiyse donguden cik
            if done:
                break
                
        # Epsilon degerini azalt
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        
        # Sonuclari kaydet
        rewards.append(total_rewards)
        epsilons.append(epsilon)
        
        # Her 1000 bolumde bir ilerlemeyi goster
        if episode % 1000 == 0:
            clear_output(wait=True)
            print(f"Bolum: {episode}")
            print(f"Ortalama Odul: {np.mean(rewards[-1000:])}")
            print(f"Epsilon: {epsilon:.2f}")
    
    return q_table, rewards, epsilons

# Q-learning algoritmasini calistir
q_table, rewards, epsilons = q_learning(env, q_table, learning_rate, gamma, epsilon, max_epsilon, min_epsilon, decay_rate, total_episodes, max_steps)

# Sonuclari gorsellestir
# Odul grafigi
plt.figure(figsize=(10, 6))
plt.plot(range(total_episodes), rewards)
plt.title('Egitim Surecindeki Odul')
plt.xlabel('Bolum')
plt.ylabel('Odul')
plt.grid(True)
plt.show()

# Epsilon grafigi
plt.figure(figsize=(10, 6))
plt.plot(range(total_episodes), epsilons)
plt.title('Epsilon Degerinin Azalmasi')
plt.xlabel('Bolum')
plt.ylabel('Epsilon')
plt.grid(True)
plt.show()

# Egitilmis ajani test et
def evaluate_agent(env, q_table, num_episodes=10):
    total_rewards = []
    
    for i in range(num_episodes):
        state = env.reset()[0]
        done = False
        episode_reward = 0
        
        print(f"Test Bolumu {i+1}")
        
        while not done:
            # En iyi eylemi sec
            action = np.argmax(q_table[state, :])
            
            # Eylemi gerceklestir
            new_state, reward, done, truncated, info = env.step(action)
            
            # Ortami gorsellestir
            env.render()
            time.sleep(0.5)  # Gorsellestirmeyi yavaslat
            
            # Durumu guncelle ve odulu topla
            state = new_state
            episode_reward += reward
            
            if done:
                print(f"Bolum Odulu: {episode_reward}")
                total_rewards.append(episode_reward)
                break
                
    print(f"Ortalama Test Odulu: {np.mean(total_rewards)}")
    
    return total_rewards

# Egitilmis ajani test et
test_rewards = evaluate_agent(env, q_table)

# Ortami kapat
env.close()`}</code>
                  </pre>
                </div>
              </TabsContent>
              
              <TabsContent value="explanation" className="p-6 m-0 space-y-4">
                <h3 className="text-xl font-bold">Kod Aciklamasi</h3>
                
                <div>
                  <h4 className="font-semibold">1. Ortam Kurulumu</h4>
                  <p className="text-sm text-muted-foreground">
                    OpenAI Gym kutuphanesinden FrozenLake-v1 ortamini kullaniyoruz. Bu ortamda, ajan buz uzerinde kaymadan hedef noktaya ulasmaya calisir. Ortam 4x4 bir izgara dunyasidir ve bazi hucreler delik icerir. Amac, baslangic noktasindan (sol ust) hedef noktasina (sag alt) guvenli bir sekilde ulasmaktir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">2. Q-Learning Algoritmasi</h4>
                  <p className="text-sm text-muted-foreground">
                    Q-learning, bir model-free reinforcement learning algoritmasidir. Her durum-eylem cifti icin bir Q degeri tutar ve bu degerler, ajanin aldigi odullere gore guncellenir. Algoritma su adimlari izler:
                    <br /><br />
                    1. Ortami sifirla ve baslangic durumunu al<br />
                    2. Her adimda:<br />
                    &nbsp;&nbsp;a. Epsilon-greedy stratejisi ile bir eylem sec (kesif veya somuru)<br />
                    &nbsp;&nbsp;b. Eylemi gerceklestir ve yeni durum, odul bilgilerini al<br />
                    &nbsp;&nbsp;c. Q-tablosunu Bellman denklemi ile guncelle<br />
                    &nbsp;&nbsp;d. Durumu guncelle ve odulu topla<br />
                    3. Bolum bittiginde epsilon degerini azalt (kesif oranini dusur)
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">3. Kesif ve Somuru Dengesi</h4>
                  <p className="text-sm text-muted-foreground">
                    Reinforcement learning&apos;de, ajan yeni durumlari kesfetme (exploration) ve bildigi en iyi eylemleri secme (exploitation) arasinda bir denge kurmalidir. Bu kodda epsilon-greedy stratejisi kullanilir:
                    <br /><br />
                    - Epsilon olasiligi ile rastgele bir eylem secilir (kesif)<br />
                    - 1-epsilon olasiligi ile Q-tablosuna gore en iyi eylem secilir (somuru)<br />
                    - Egitim ilerledikce epsilon degeri azalir, boylece ajan zamanla daha az kesif, daha cok somuru yapar
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">4. Sonuclarin Degerlendirilmesi</h4>
                  <p className="text-sm text-muted-foreground">
                    Egitim tamamlandiktan sonra, ajanin performansini degerlendirmek icin iki grafik cizilir:
                    <br /><br />
                    - Odul grafigi: Her bolumde alinan toplam odulu gosterir<br />
                    - Epsilon grafigi: Epsilon degerinin zamanla nasil azaldigini gosterir<br />
                    <br />
                    Ayrica, egitilmis ajan test edilerek gercek performansi gozlemlenir. Test sirasinda, ajan artik kesif yapmaz ve her zaman Q-tablosuna gore en iyi eylemi secer.
                  </p>
                </div>
              </TabsContent>
              
              <TabsContent value="output" className="p-6 m-0 space-y-4">
                <h3 className="text-xl font-bold">Cikti Ornekleri</h3>
                
                <div>
                  <h4 className="font-semibold">Egitim Ilerlemesi</h4>
                  <pre className="bg-gray-100 dark:bg-gray-850 p-4 rounded-md overflow-x-auto text-sm">
                    {`Bolum: 14000
Ortalama Odul: 0.642
Epsilon: 0.01`}
                  </pre>
                  <p className="text-sm text-muted-foreground mt-2">
                    Egitim sirasinda her 1000 bolumde bir gosterilen ilerleme. Ortalama odul zamanla artar ve epsilon degeri azalir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">Odul ve Epsilon Grafikleri</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-gray-100 dark:bg-gray-850 p-4 rounded-md">
                      <Image 
                        src="/images/code-examples/rl-reward-graph.jpg" 
                        alt="Egitim Surecindeki Odul" 
                        width={400} 
                        height={300} 
                        className="mx-auto"
                      />
                      <p className="text-sm text-muted-foreground mt-2 text-center">
                        Egitim surecinde alinan odullerin grafigi
                      </p>
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-850 p-4 rounded-md">
                      <Image 
                        src="/images/code-examples/rl-epsilon-graph.jpg" 
                        alt="Epsilon Degerinin Azalmasi" 
                        width={400} 
                        height={300} 
                        className="mx-auto"
                      />
                      <p className="text-sm text-muted-foreground mt-2 text-center">
                        Epsilon degerinin zamanla azalmasi
                      </p>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold">FrozenLake Ortami Gorsellestirmesi</h4>
                  <div className="bg-gray-100 dark:bg-gray-850 p-4 rounded-md">
                    <pre className="text-sm">
                      {`
  (Start) S F F F
         F H F H
         F F F H
         H F G (Goal)
      `}
                    </pre>
                    <p className="text-sm text-muted-foreground mt-2 text-center">
                      FrozenLake ortaminin temsili gorunumu. S: Baslangic, G: Hedef, F: Donmus gol (guvenli), H: Delik (tehlikeli)
                    </p>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold">Test Sonuclari</h4>
                  <pre className="bg-gray-100 dark:bg-gray-850 p-4 rounded-md overflow-x-auto text-sm">
                    {`Test Bolumu 1
Bolum Odulu: 1.0
Test Bolumu 2
Bolum Odulu: 1.0
Test Bolumu 3
Bolum Odulu: 0.0
Test Bolumu 4
Bolum Odulu: 1.0
Test Bolumu 5
Bolum Odulu: 1.0
Ortalama Test Odulu: 0.8`}
                  </pre>
                  <p className="text-sm text-muted-foreground mt-2">
                    Egitilmis ajanin test sonuclari. Odul 1.0, ajanin hedefe ulastigini; 0.0 ise delige dustugunu gosterir.
                  </p>
                </div>
              </TabsContent>
            </Tabs>
          </div>
          
          <div className="mt-8">
            <h3 className="text-xl font-bold mb-4">Ek Kaynaklar</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">OpenAI Gym Dokumantasyonu</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Reinforcement learning ortamlari icin OpenAI Gym kutuphanesinin resmi dokumantasyonu.
                  </p>
                </CardContent>
                <CardFooter>
                  <Button asChild variant="outline" className="w-full">
                    <a href="https://gymnasium.farama.org/" target="_blank" rel="noopener noreferrer">
                      Ziyaret Et
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </a>
                  </Button>
                </CardFooter>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Reinforcement Learning: Giris</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Richard S. Sutton ve Andrew G. Barto&apos;nun reinforcement learning hakkinda temel kitabi.
                  </p>
                </CardContent>
                <CardFooter>
                  <Button asChild variant="outline" className="w-full">
                    <a href="http://incompleteideas.net/book/the-book-2nd.html" target="_blank" rel="noopener noreferrer">
                      Ziyaret Et
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </a>
                  </Button>
                </CardFooter>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
