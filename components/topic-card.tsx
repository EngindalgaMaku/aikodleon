import Link from 'next/link'

interface TopicCardProps {
  title: string
  description: string
  href: string
}

export function TopicCard({ title, description, href }: TopicCardProps) {
  return (
    <Link
      href={href}
      className="block p-6 bg-card hover:bg-card/90 rounded-lg transition-colors"
    >
      <article>
        <h3 className="text-2xl font-semibold mb-4">{title}</h3>
        <p className="text-muted-foreground">{description}</p>
      </article>
    </Link>
  )
} 