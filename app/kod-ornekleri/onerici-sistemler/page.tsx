import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, Download, Github, Copy } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export const metadata: Metadata = {
  title: 'Önerici Sistemler | Kod Örnekleri | Kodleon',
  description: 'Scikit-learn kullanarak işbirlikçi filtreleme tabanlı basit bir film öneri sistemi oluşturma.',
  openGraph: {
    title: 'Önerici Sistemler | Kodleon',
    description: 'Scikit-learn kullanarak işbirlikçi filtreleme tabanlı basit bir film öneri sistemi oluşturma.',
    images: [{ url: '/images/code-examples/recommender-system.jpg' }], // Bu resmin eklenmesi gerekiyor
  },
};

export default function RecommenderSystemsPage() {
  return (
    <div className="container max-w-6xl mx-auto py-12 px-4">
      <div className="mb-8">
        <Button asChild variant="outline" size="sm" className="gap-1">
          <Link href="/kod-ornekleri">
            <ArrowLeft className="h-4 w-4" />
            Tüm Kod Örnekleri
          </Link>
        </Button>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Sol Taraf - Açıklama */}
        <div className="lg:col-span-1">
          <div className="sticky top-20">
            <h1 className="text-3xl font-bold mb-4">Önerici Sistemler</h1>
            
            <div className="flex items-center gap-2 mb-4">
               <span className="px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                Makine Öğrenmesi
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
                İleri
              </span>
            </div>
            
            <p className="text-muted-foreground mb-6">
             Bu örnekte, `scikit-learn` ve `pandas` kütüphanelerini kullanarak, kullanıcıların filmlere verdiği oylara dayanan basit bir işbirlikçi filtreleme (collaborative filtering) öneri sistemi oluşturacaksınız.
            </p>
            
            <div className="space-y-4 mb-6">
              <div>
                <h3 className="font-medium">Gereksinimler:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Python 3.7+</li>
                  <li>Pandas</li>
                  <li>Scikit-learn</li>
                </ul>
              </div>
              
              <div>
                <h3 className="font-medium">Öğrenilecek Kavramlar:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Öneri sistemleri temelleri</li>
                  <li>İşbirlikçi Filtreleme (Collaborative Filtering)</li>
                  <li>Kullanıcı-Öğe Matrisi</li>
                  <li>Kosinüs Benzerliği</li>
                </ul>
              </div>
            </div>
            
            <div className="flex flex-col gap-2">
               <Button asChild variant="default" className="gap-2" disabled>
                <a href="#">
                  <Download className="h-4 w-4" />
                  Jupyter Notebook (Yakında)
                </a>
              </Button>
              <Button asChild variant="outline" className="gap-2" disabled>
                <a href="#" target="_blank" rel="noopener noreferrer">
                  <Github className="h-4 w-4" />
                  GitHub'da Görüntüle (Yakında)
                </a>
              </Button>
            </div>
          </div>
        </div>
        
        {/* Sağ Taraf - Kod */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-850 rounded-xl shadow-md overflow-hidden">
            <Tabs defaultValue="code" className="w-full">
              <div className="border-b">
                <TabsList className="p-0 bg-transparent">
                  <TabsTrigger value="code" className="rounded-none py-3 px-6 data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">
                    Kod
                  </TabsTrigger>
                  <TabsTrigger value="explanation" className="rounded-none py-3 px-6 data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">
                    Açıklama
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
                    <code>{`import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Örnek veri seti: Kullanıcıların filmlere verdiği puanlar (1-5)
data = {
    'Kullanici1': {'Film A': 5, 'Film B': 3, 'Film C': 4, 'Film D': 0},
    'Kullanici2': {'Film A': 4, 'Film B': 0, 'Film C': 5, 'Film D': 2},
    'Kullanici3': {'Film A': 2, 'Film B': 5, 'Film C': 0, 'Film D': 5},
    'Kullanici4': {'Film A': 0, 'Film B': 4, 'Film C': 2, 'Film D': 4},
}

df = pd.DataFrame(data).fillna(0) # Puanlanmamış filmleri 0 ile doldur

# Kullanıcılar arası benzerliği hesapla (Kosinüs Benzerliği)
user_similarity = cosine_similarity(df.T)
user_similarity_df = pd.DataFrame(user_similarity, index=df.columns, columns=df.columns)

print("Kullanıcı Benzerlik Matrisi:")
print(user_similarity_df)

# Bir kullanıcıya film önerme fonksiyonu
def recommend_movies(user, num_recommendations=1):
    # Benzerliği en yüksek kullanıcıyı bul (kendisi hariç)
    similar_users = user_similarity_df[user].sort_values(ascending=False)
    most_similar_user = similar_users.index[1]
    
    # Hedef kullanıcının izlemediği ama benzer kullanıcının yüksek puan verdiği filmleri bul
    user_movies = df[user][df[user] > 0].index
    similar_user_movies = df[most_similar_user][df[most_similar_user] > 0].index
    
    recommendations = list(set(similar_user_movies) - set(user_movies))
    
    return recommendations[:num_recommendations]

# 'Kullanici1' için film öner
recommendations_for_user1 = recommend_movies('Kullanici1')
print(f"\\nKullanici1 için önerilen film(ler): {recommendations_for_user1}")
`}</code>
                  </pre>
                </div>
              </TabsContent>
              
              <TabsContent value="explanation" className="p-6 m-0 space-y-4">
                <h3 className="text-xl font-bold">Kod Açıklaması</h3>
                
                <div>
                  <h4 className="font-semibold">1. Veri Seti Oluşturma</h4>
                  <p className="text-sm text-muted-foreground">
                    Örnek bir kullanıcı-film puan matrisi oluşturulur. Gerçek dünyada bu veri, bir veritabanından veya dosyadan okunur. `fillna(0)` ile kullanıcıların puanlamadığı filmler 0 olarak işaretlenir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">2. Benzerlik Hesaplama</h4>
                  <p className="text-sm text-muted-foreground">
                    `cosine_similarity` fonksiyonu kullanılarak kullanıcıların puanlama vektörleri arasındaki benzerlik hesaplanır. Bu, hangi kullanıcıların zevklerinin birbirine benzediğini gösteren bir matris oluşturur.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">3. Öneri Fonksiyonu</h4>
                  <p className="text-sm text-muted-foreground">
                    `recommend_movies` fonksiyonu, hedef kullanıcıya en çok benzeyen kullanıcıyı bulur. Ardından, benzer kullanıcının izleyip yüksek puan verdiği ancak hedef kullanıcının henüz izlemediği filmleri bularak öneri olarak sunar.
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold">4. İşbirlikçi Filtreleme (Collaborative Filtering)</h4>
                  <p className="text-sm text-muted-foreground">
                    Bu yaklaşımın temel mantığı şudur: "Sizin zevklerinize benzer kişilerin beğendiği şeyleri muhtemelen siz de beğenirsiniz." Kod, bu mantığı uygulayarak kullanıcılar arası benzerliğe dayalı öneriler yapar.
                  </p>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  );
} 