'use client';

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Github, ExternalLink } from "lucide-react";
import Link from "next/link";

export default function OOPProjects() {
  const projects = [
    {
      title: "Kütüphane Yönetim Sistemi",
      description: "Kitapları, üyeleri ve ödünç alma işlemlerini yöneten bir kütüphane sistemi geliştirin.",
      difficulty: "Orta",
      skills: ["Sınıflar", "Kalıtım", "Dosya İşlemleri"],
      requirements: [
        "Kitap ve Üye sınıfları oluşturma",
        "Ödünç alma ve iade işlemleri",
        "Verileri dosyada saklama",
        "Arama ve filtreleme özellikleri"
      ],
      githubLink: "#"
    },
    {
      title: "Banka Hesap Yönetimi",
      description: "Farklı türde banka hesaplarını ve işlemlerini yöneten bir sistem oluşturun.",
      difficulty: "Başlangıç",
      skills: ["Sınıflar", "Kapsülleme", "Temel İşlemler"],
      requirements: [
        "Hesap sınıfı oluşturma",
        "Para yatırma ve çekme işlemleri",
        "Bakiye kontrolü",
        "İşlem geçmişi tutma"
      ],
      githubLink: "#"
    },
    {
      title: "Online Alışveriş Sistemi",
      description: "Ürünleri, sepeti ve siparişleri yöneten bir e-ticaret sistemi geliştirin.",
      difficulty: "İleri",
      skills: ["Sınıflar", "Kalıtım", "Çok Biçimlilik", "Hata Yönetimi"],
      requirements: [
        "Ürün ve Kategori sınıfları",
        "Alışveriş sepeti yönetimi",
        "Sipariş işleme sistemi",
        "Stok kontrolü"
      ],
      githubLink: "#"
    }
  ];

  return (
    <div className="container max-w-4xl mx-auto py-8 px-4">
      <div className="mb-8">
        <Link 
          href="/topics/python/nesne-tabanli-programlama" 
          className="inline-flex items-center text-primary hover:underline mb-4"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Nesne Tabanlı Programlama'ya Dön
        </Link>
        <h1 className="text-3xl font-bold mb-2">Projeler</h1>
        <p className="text-muted-foreground">
          Nesne tabanlı programlama becerilerinizi geliştirmek için pratik projeler.
        </p>
      </div>

      <div className="grid gap-6">
        {projects.map((project, index) => (
          <Card key={index}>
            <CardHeader>
              <div className="flex justify-between items-start">
                <div>
                  <CardTitle className="text-xl mb-1">{project.title}</CardTitle>
                  <CardDescription>{project.description}</CardDescription>
                </div>
                <span className={`px-3 py-1 rounded-full text-sm font-medium
                  ${project.difficulty === 'Başlangıç' ? 'bg-green-100 text-green-800' :
                    project.difficulty === 'Orta' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'}`}>
                  {project.difficulty}
                </span>
              </div>
            </CardHeader>
            <CardContent>
              <div className="mb-4">
                <h3 className="font-semibold mb-2">Gerekli Beceriler:</h3>
                <div className="flex flex-wrap gap-2">
                  {project.skills.map((skill, skillIndex) => (
                    <span 
                      key={skillIndex}
                      className="bg-primary/10 text-primary px-2 py-1 rounded-md text-sm"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Gereksinimler:</h3>
                <ul className="list-disc list-inside text-muted-foreground">
                  {project.requirements.map((req, reqIndex) => (
                    <li key={reqIndex}>{req}</li>
                  ))}
                </ul>
              </div>
            </CardContent>
            <CardFooter>
              <Button asChild variant="outline" className="gap-2">
                <Link href={project.githubLink}>
                  <Github className="h-4 w-4" />
                  Proje Detayları
                  <ExternalLink className="h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
} 