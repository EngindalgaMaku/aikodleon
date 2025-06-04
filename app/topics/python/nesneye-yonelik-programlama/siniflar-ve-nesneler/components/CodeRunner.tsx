'use client';

import { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Loader2, Play, RotateCcw } from 'lucide-react';

interface CodeRunnerProps {
  initialCode: string;
}

export default function CodeRunner({ initialCode }: CodeRunnerProps) {
  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [pyodide, setPyodide] = useState<any>(null);
  const editorRef = useRef<any>(null);

  useEffect(() => {
    // Load CodeMirror
    import('codemirror').then(async (CodeMirror) => {
      // Load Python mode and theme
      await import('codemirror/mode/python/python');
      await import('codemirror/theme/monokai.css');
      await import('codemirror/lib/codemirror.css');

      if (!editorRef.current) {
        const editor = CodeMirror.default(document.getElementById('code-editor')!, {
          value: code,
          mode: 'python',
          theme: 'monokai',
          lineNumbers: true,
          indentUnit: 4,
          extraKeys: { 'Tab': 'indentMore', 'Shift-Tab': 'indentLess' }
        });

        editor.on('change', (cm) => {
          setCode(cm.getValue());
        });

        editorRef.current = editor;
      }
    });

    // Load Pyodide
    const loadPyodide = async () => {
      try {
        setIsLoading(true);
        const { loadPyodide } = await import("pyodide");
        const pyodide = await loadPyodide({
          indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"
        });
        setPyodide(pyodide);
      } catch (error) {
        console.error('Pyodide yüklenirken hata:', error);
        setOutput('Python çalışma ortamı yüklenirken bir hata oluştu.');
      } finally {
        setIsLoading(false);
      }
    };

    loadPyodide();
  }, []);

  const runCode = async () => {
    if (!pyodide) {
      setOutput('Python çalışma ortamı henüz yüklenmedi. Lütfen bekleyin...');
      return;
    }

    setIsLoading(true);
    setOutput('');

    try {
      // Redirect stdout to capture print statements
      pyodide.runPython(`
        import sys
        from io import StringIO
        sys.stdout = StringIO()
      `);

      // Run the actual code
      await pyodide.runPythonAsync(code);

      // Get the captured output
      const stdout = pyodide.runPython("sys.stdout.getvalue()");
      setOutput(stdout);

      // Reset stdout
      pyodide.runPython("sys.stdout = sys.__stdout__");
    } catch (error: any) {
      setOutput(`Hata: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const resetCode = () => {
    setCode(initialCode);
    if (editorRef.current) {
      editorRef.current.setValue(initialCode);
    }
    setOutput('');
  };

  return (
    <div className="border rounded-lg overflow-hidden">
      <div className="bg-gray-800 p-4 flex justify-between items-center">
        <h3 className="text-white font-medium">Python Kod Editörü</h3>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={resetCode}
            disabled={isLoading}
          >
            <RotateCcw className="h-4 w-4 mr-1" />
            Sıfırla
          </Button>
          <Button
            variant="default"
            size="sm"
            onClick={runCode}
            disabled={isLoading}
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 mr-1 animate-spin" />
            ) : (
              <Play className="h-4 w-4 mr-1" />
            )}
            Çalıştır
          </Button>
        </div>
      </div>
      
      <div id="code-editor" className="min-h-[200px]" />
      
      {output && (
        <div className="bg-gray-100 p-4 border-t">
          <h4 className="font-medium mb-2">Çıktı:</h4>
          <pre className="whitespace-pre-wrap">{output}</pre>
        </div>
      )}
    </div>
  );
} 