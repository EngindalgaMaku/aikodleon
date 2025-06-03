import { Metadata } from "next";
import HomePageClientContent from "@/components/HomePageClientContent";

export const metadata: Metadata = {
  title: "Kodleon | Türkiye'nin Lider Yapay Zeka Eğitim Platformu",
  description: "Türkiye'nin lider yapay zeka eğitim platformu Kodleon ile en güncel AI becerilerini kazanın.",
};

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col">
      <HomePageClientContent />
    </main>
  );
}