import React from 'react';
import { Link } from 'react-router-dom';
import { Heart, LineChart, User } from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';

const Home = () => {
  return (
    <div className="space-y-10 pb-8">
      <div className="space-y-4 text-center">
        <h1 className="text-3xl font-bold tracking-tighter sm:text-5xl">
          Heart Disease Prediction System
        </h1>
        <p className="mx-auto max-w-[700px] text-gray-500 md:text-xl/relaxed">
          An advanced machine learning application designed to assess heart disease risk factors
          using GPU-accelerated prediction models.
        </p>
      </div>

      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        <Card className="border-none shadow-md">
          <CardHeader className="space-y-2">
            <div className="flex items-center justify-center h-12 w-12 rounded-full bg-primary/10">
              <Heart className="h-6 w-6 text-primary" />
            </div>
            <CardTitle>Risk Assessment</CardTitle>
            <CardDescription>
              Get instant heart disease risk assessment based on your health parameters
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Our model has been trained on thousands of cases to provide accurate risk predictions.
              Simply enter your health data and get an immediate assessment.
            </p>
          </CardContent>
          <CardFooter>
            <Link to="/predict" className="w-full">
              <Button className="w-full">Start Assessment</Button>
            </Link>
          </CardFooter>
        </Card>

        <Card className="border-none shadow-md">
          <CardHeader className="space-y-2">
            <div className="flex items-center justify-center h-12 w-12 rounded-full bg-primary/10">
              <LineChart className="h-6 w-6 text-primary" />
            </div>
            <CardTitle>Model Performance</CardTitle>
            <CardDescription>
              Explore the performance metrics of our different prediction models
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              We've utilized GPU-accelerated computing to build and compare multiple machine learning models. 
              See detailed metrics like accuracy, precision, recall, and AUC scores.
            </p>
          </CardContent>
          <CardFooter>
            <Link to="/metrics" className="w-full">
              <Button variant="outline" className="w-full">View Metrics</Button>
            </Link>
          </CardFooter>
        </Card>

        <Card className="border-none shadow-md">
          <CardHeader className="space-y-2">
            <div className="flex items-center justify-center h-12 w-12 rounded-full bg-primary/10">
              <User className="h-6 w-6 text-primary" />
            </div>
            <CardTitle>Healthcare Tool</CardTitle>
            <CardDescription>
              A supporting tool for healthcare professionals and individuals
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              This application is designed as an educational tool to demonstrate how machine learning 
              can be applied to healthcare risk assessment. Always consult healthcare professionals for diagnosis.
            </p>
          </CardContent>
          <CardFooter>
            <Link to="/predict" className="w-full">
              <Button variant="outline" className="w-full">Learn More</Button>
            </Link>
          </CardFooter>
        </Card>
      </div>

      <div className="rounded-lg border bg-card p-8 text-center shadow">
        <h2 className="text-2xl font-bold">About This Project</h2>
        <p className="mt-4 text-muted-foreground">
          This heart disease prediction system utilizes Big Data Analytics techniques to process health data 
          and predict heart disease risk. We've implemented multiple machine learning models using GPU acceleration 
          for optimal performance, and built a modern web interface for easy interaction with the prediction system.
        </p>
        <p className="mt-4 text-sm text-muted-foreground">
          <strong>Disclaimer:</strong> This application is for educational and research purposes only and is not intended 
          to be a substitute for professional medical advice, diagnosis, or treatment.
        </p>
      </div>
    </div>
  );
};

export default Home;
