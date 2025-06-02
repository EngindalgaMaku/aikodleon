import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

const topicsDirectory = path.join(process.cwd(), 'topics');

export interface MarkdownFrontmatter {
  title?: string;
  description?: string;
  date?: string;
  [key: string]: any; // For any other custom frontmatter fields
}

export function getMarkdownContent(filePath: string): { rawContent: string; frontmatter: MarkdownFrontmatter } {
  const correctedFilePath = filePath.endsWith('.md') ? filePath : `${filePath}.md`;
  const fullPath = path.join(topicsDirectory, correctedFilePath);
  
  let fileContents;
  try {
    fileContents = fs.readFileSync(fullPath, 'utf8');
  } catch (err: any) {
    console.error(`Error reading markdown file at ${fullPath}: ${err.message}`);
    return {
      rawContent: `İçerik bu konumda bulunamadı veya yüklenemedi: \`topics/${correctedFilePath}\`. Lütfen dosya yolunu kontrol edin.`,
      frontmatter: { title: 'İçerik Bulunamadı' },
    };
  }

  const matterResult = matter(fileContents);

  return {
    rawContent: matterResult.content,
    frontmatter: matterResult.data as MarkdownFrontmatter,
  };
}

// Opsiyonel: Belirli bir dizindeki tüm markdown dosyalarının yollarını almak için bir fonksiyon
// Bu, örneğin blog veya dokümantasyon listeleme sayfaları için kullanılabilir.
export function getAllMarkdownSlugsInDirectory(directoryPath: string): string[] {
  const fullDirectoryPath = path.join(topicsDirectory, directoryPath);
  try {
    const fileNames = fs.readdirSync(fullDirectoryPath);
    return fileNames
      .filter(fileName => fileName.endsWith('.md'))
      .map(fileName => fileName.replace(/\.md$/, ''));
  } catch (error) {
    console.error(`Error reading directory ${fullDirectoryPath}:`, error);
    return [];
  }
} 