import BlogPage from "./client";
import { getAllPosts } from "@/lib/mdx";

export const revalidate = 3600; // Revalidate every hour

export default async function Page() {
  const posts = await getAllPosts();
  return <BlogPage posts={posts} />;
}