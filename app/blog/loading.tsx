import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export default function BlogLoading() {
  return (
    <div className="container max-w-6xl mx-auto py-12">
      <div className="max-w-3xl mx-auto text-center mb-12">
        <Skeleton className="h-10 w-[200px] mx-auto mb-4" />
        <Skeleton className="h-6 w-[400px] mx-auto" />
        <Skeleton className="h-4 w-[100px] mx-auto mt-2" />
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {Array.from({ length: 6 }).map((_, index) => (
          <Card key={index} className="overflow-hidden">
            <Skeleton className="h-48 w-full" />
            <CardHeader>
              <Skeleton className="h-6 w-[250px] mb-2" />
              <Skeleton className="h-4 w-[350px]" />
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <Skeleton className="h-4 w-[100px]" />
                <Skeleton className="h-4 w-[150px]" />
              </div>
            </CardContent>
            <CardFooter>
              <Skeleton className="h-9 w-[120px] ml-auto" />
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
} 