import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

const appPostsDirectory = path.join(process.cwd(), 'app/blog');
const rootPostsDirectory = path.join(process.cwd(), 'blog');
const topicsDirectory = path.join(process.cwd(), 'topics');

// Define interfaces
export interface BlogPost {
  slug: string;
  title: string;
  description: string;
  date: string;
  author: string;
  category: string;
  tags: string[];
  image?: string;
  content?: string;
  topicPath?: string | null;
  isTypescriptPage?: boolean;
}

function getPostFromPath(fullPath: string, slug: string): BlogPost {
  const fileContents = fs.readFileSync(fullPath, 'utf8');
  const { data } = matter(fileContents);

  return {
    slug,
    ...(data as { 
      title: string;
      description: string;
      date: string;
      author: string;
      category: string;
      tags: string[];
      image: string;
    })
  };
}

function extractMetadataFromTSX(content: string): BlogPost | null {
  // First try to match the metadata object directly
  const metadataMatch = content.match(/export\s+const\s+metadata\s*=\s*({[\s\S]*?}\s*);/);
  if (!metadataMatch) return null;

  try {
    // Clean up the metadata string
    let metadataStr = metadataMatch[1]
      .replace(/createPageMetadata\s*\(\s*/, '') // Remove createPageMetadata(
      .replace(/\)\s*$/, '') // Remove trailing )
      .trim();
    
    // If it's still wrapped in createPageMetadata, extract the object
    if (metadataStr.startsWith('createPageMetadata')) {
      const innerMatch = metadataStr.match(/createPageMetadata\s*\(\s*({[\s\S]*?})\s*\)/);
      if (innerMatch) {
        metadataStr = innerMatch[1];
      }
    }

    // Evaluate the cleaned metadata string
    // Replace any remaining function calls with their argument
    metadataStr = metadataStr.replace(/\w+\((.*)\)/g, '$1');
    
    // Parse the metadata
    const metadata = eval(`(${metadataStr})`);
    
    return {
      slug: '', // Will be set by the calling code
      title: metadata.title,
      description: metadata.description,
      date: metadata.date || new Date().toISOString(),
      author: metadata.author || 'Kodleon Ekibi',
      category: metadata.category || 'Teknoloji',
      tags: metadata.keywords || [],
      image: metadata.openGraph?.images?.[0]?.url || '/images/blog-default.jpg'
    };
  } catch (error) {
    console.error('Error parsing TSX metadata:', error);
    return null;
  }
}

function getPostsFromDirectory(directory: string, isTopicDir = false): BlogPost[] {
  if (!fs.existsSync(directory)) {
    return [];
  }
  
  const entries = fs.readdirSync(directory, { withFileTypes: true });
  return entries.flatMap<BlogPost>(entry => {
    // Skip the [slug] directory and page.tsx
    if (entry.name === '[slug]' || entry.name === 'page.tsx') {
      return [];
    }

    // If it's a .mdx or .md file in the root
    if (entry.isFile() && (entry.name.endsWith('.mdx') || entry.name.endsWith('.md'))) {
      const slug = entry.name.replace(/\.(mdx|md)$/, '');
      const fullPath = path.join(directory, entry.name);
      
      try {
        const fileContents = fs.readFileSync(fullPath, 'utf8');
        const { data } = matter(fileContents);
        
        // For topic directories, check if file has frontmatter data
        if (isTopicDir) {
          if (!data.title) {
            // Skip files without proper frontmatter in topic directories
            return [];
          }
          
          // Add default category based on directory for topic posts
          if (!data.category) {
            const dirName = path.basename(directory);
            data.category = dirName.split('-').map(word => 
              word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');
          }
          
          // Add default date if missing
          if (!data.date) {
            data.date = new Date().toISOString();
          }
        }
        
        // Ensure all required fields are present
        const post: BlogPost = {
          slug,
          title: data.title || '',
          description: data.description || '',
          date: data.date || new Date().toISOString(),
          author: data.author || 'Kodleon Ekibi',
          category: data.category || 'Genel',
          tags: data.tags || [],
          topicPath: isTopicDir ? directory.replace(process.cwd(), '').replace(/\\/g, '/') : null,
          image: data.image
        };
        
        return [post];
      } catch (error) {
        console.error(`Error processing file ${fullPath}:`, error);
        return [];
      }
    }
    
    // If it's a directory
    if (entry.isDirectory()) {
      const dirPath = path.join(directory, entry.name);
      
      // If we're in the topics directory, recursively scan subdirectories
      if (isTopicDir) {
        return getPostsFromDirectory(dirPath, true);
      }
      
      // Otherwise, look for specific files in the directory
      try {
        const dirEntries = fs.readdirSync(dirPath);
        
        // Check for index.md first
        if (dirEntries.includes('index.md')) {
          const fullPath = path.join(dirPath, 'index.md');
          return [getPostFromPath(fullPath, entry.name)];
        }
        
        // Then check for page.mdx
        if (dirEntries.includes('page.mdx')) {
          const fullPath = path.join(dirPath, 'page.mdx');
          return [getPostFromPath(fullPath, entry.name)];
        }
        
        // Then check for page.tsx
        if (dirEntries.includes('page.tsx')) {
          const fullPath = path.join(dirPath, 'page.tsx');
          const fileContents = fs.readFileSync(fullPath, 'utf8');
          const metadata = extractMetadataFromTSX(fileContents);
          
          if (metadata) {
            const post: BlogPost = {
              ...metadata,
              slug: entry.name
            };
            return [post];
          }
        }
      } catch (error) {
        console.error(`Error processing directory ${dirPath}:`, error);
        return [];
      }
    }
    return [];
  });
}

function getTopicPosts(): BlogPost[] {
  if (!fs.existsSync(topicsDirectory)) {
    return [];
  }
  
  // Get all markdown files from topics directory and subdirectories
  return getPostsFromDirectory(topicsDirectory, true);
}

export async function getAllPosts(): Promise<BlogPost[]> {
  // Get posts from all directories
  const appPosts = getPostsFromDirectory(appPostsDirectory);
  const rootPosts = getPostsFromDirectory(rootPostsDirectory);
  const topicPosts = getTopicPosts();
  
  // Combine all posts
  const allPostsData = [...appPosts, ...rootPosts, ...topicPosts];

  // Sort posts by date
  return allPostsData.sort((a, b) => {
    if (a.date < b.date) {
      return 1;
    } else {
      return -1;
    }
  });
}

export async function getPostBySlug(slug: string): Promise<BlogPost> {
  // First try as a direct MDX file in app directory
  const appMdxPath = path.join(appPostsDirectory, `${slug}.mdx`);
  if (fs.existsSync(appMdxPath)) {
    const fileContents = fs.readFileSync(appMdxPath, 'utf8');
    const { data, content } = matter(fileContents);
    return {
      slug,
      content,
      ...(data as {
        title: string;
        description: string;
        date: string;
        author: string;
        category: string;
        tags: string[];
        image: string;
      })
    };
  }

  // Then try as a directory with page.mdx in app directory
  const appDirMdxPath = path.join(appPostsDirectory, slug, 'page.mdx');
  if (fs.existsSync(appDirMdxPath)) {
    const fileContents = fs.readFileSync(appDirMdxPath, 'utf8');
    const { data, content } = matter(fileContents);
    return {
      slug,
      content,
      ...(data as {
        title: string;
        description: string;
        date: string;
        author: string;
        category: string;
        tags: string[];
        image: string;
      })
    };
  }

  // Then try as a directory with page.tsx in app directory
  const appDirTsxPath = path.join(appPostsDirectory, slug, 'page.tsx');
  if (fs.existsSync(appDirTsxPath)) {
    const fileContents = fs.readFileSync(appDirTsxPath, 'utf8');
    const metadata = extractMetadataFromTSX(fileContents);
    if (!metadata) {
      throw new Error(`Could not extract metadata from TSX file: ${slug}`);
    }
    
    // Create a copy of metadata without the slug property
    const { slug: _, ...metadataWithoutSlug } = metadata;
    
    return {
      slug,
      ...metadataWithoutSlug,
      isTypescriptPage: true
    };
  }
  
  // Check in root blog directory
  // Try as a directory with index.md
  const rootDirIndexPath = path.join(rootPostsDirectory, slug, 'index.md');
  if (fs.existsSync(rootDirIndexPath)) {
    const fileContents = fs.readFileSync(rootDirIndexPath, 'utf8');
    const { data, content } = matter(fileContents);
    return {
      slug,
      content,
      ...(data as {
        title: string;
        description: string;
        date: string;
        author: string;
        category: string;
        tags: string[];
        image: string;
      })
    };
  }
  
  // Finally, check in topics directory and its subdirectories
  // This is a more expensive search since we need to walk the directory tree
  const findTopicFile = (dir: string): string | null => {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      if (entry.isFile() && (entry.name === `${slug}.md` || entry.name === `${slug}.mdx`)) {
        return path.join(dir, entry.name);
      }
      
      if (entry.isDirectory()) {
        const result = findTopicFile(path.join(dir, entry.name));
        if (result) return result;
      }
    }
    
    return null;
  };
  
  const topicFilePath = findTopicFile(topicsDirectory);
  if (topicFilePath) {
    const fileContents = fs.readFileSync(topicFilePath, 'utf8');
    const { data, content } = matter(fileContents);
    return {
      slug,
      content,
      topicPath: path.dirname(topicFilePath).replace(process.cwd(), '').replace(/\\/g, '/'),
      ...(data as {
        title: string;
        description: string;
        date: string;
        author: string;
        category: string;
        tags: string[];
        image: string;
      })
    };
  }

  throw new Error(`Post not found: ${slug}`);
} 