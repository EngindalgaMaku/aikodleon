import { Mail, MapPin, Phone } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

export default function ContactPage() {
  return (
    <div className="container py-12">
      <div className="max-w-3xl mx-auto text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-4">İletişime Geçin</h1>
        <p className="text-xl text-muted-foreground">
          Sorularınız için bizimle iletişime geçebilir veya özel eğitim taleplerinizi iletebilirsiniz.
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
        <Card>
          <CardContent className="flex flex-col items-center text-center p-6">
            <div className="p-3 rounded-full bg-primary/10 mb-4">
              <Mail className="h-6 w-6 text-primary" />
            </div>
            <h3 className="font-medium mb-2">Email</h3>
            <p className="text-sm text-muted-foreground">info@aiegitim.com</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="flex flex-col items-center text-center p-6">
            <div className="p-3 rounded-full bg-primary/10 mb-4">
              <Phone className="h-6 w-6 text-primary" />
            </div>
            <h3 className="font-medium mb-2">Telefon</h3>
            <p className="text-sm text-muted-foreground">+90 (212) 555 0123</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="flex flex-col items-center text-center p-6">
            <div className="p-3 rounded-full bg-primary/10 mb-4">
              <MapPin className="h-6 w-6 text-primary" />
            </div>
            <h3 className="font-medium mb-2">Adres</h3>
            <p className="text-sm text-muted-foreground">
              Levent, İstanbul
            </p>
          </CardContent>
        </Card>
      </div>
      
      <Card className="max-w-2xl mx-auto">
        <CardContent className="p-6">
          <form className="space-y-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label htmlFor="name" className="text-sm font-medium">
                  İsim
                </label>
                <Input id="name" placeholder="İsminizi girin" />
              </div>
              <div className="space-y-2">
                <label htmlFor="email" className="text-sm font-medium">
                  Email
                </label>
                <Input id="email" type="email" placeholder="Email adresinizi girin" />
              </div>
            </div>
            
            <div className="space-y-2">
              <label htmlFor="subject" className="text-sm font-medium">
                Konu
              </label>
              <Input id="subject" placeholder="Mesajınızın konusu" />
            </div>
            
            <div className="space-y-2">
              <label htmlFor="message" className="text-sm font-medium">
                Mesaj
              </label>
              <Textarea
                id="message"
                placeholder="Mesajınızı yazın"
                rows={6}
              />
            </div>
            
            <Button className="w-full">Gönder</Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}