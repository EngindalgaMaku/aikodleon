'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Play } from 'lucide-react';

interface CodeRunnerProps {
  initialCode: string;
}

export default function CodeRunner({ initialCode }: CodeRunnerProps) {
  const [code] = useState(initialCode);

  const openInPlayground = () => {
    // Python Tutor URL'sini oluştur
    const encodedCode = encodeURIComponent(code);
    const playgroundUrl = `https://pythontutor.com/visualize.html#code=${encodedCode}&cumulative=false&heapPrimitives=true&mode=edit&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false`;
    
    // Yeni pencerede aç
    window.open(playgroundUrl, '_blank');
  };

  return (
    <div className="border rounded-lg overflow-hidden">
      <div className="bg-gray-800 p-4 flex justify-between items-center">
        <h3 className="text-white font-medium">Python Kod Örneği</h3>
        <Button
          variant="default"
          size="sm"
          onClick={openInPlayground}
          className="bg-green-600 hover:bg-green-700"
        >
          <Play className="h-4 w-4 mr-1" />
          Python Tutor'da Çalıştır
        </Button>
      </div>
      
      <pre className="bg-[#282c34] p-4 m-0 overflow-x-auto">
        <code className="text-white whitespace-pre-wrap">{code}</code>
      </pre>
    </div>
  );
} 