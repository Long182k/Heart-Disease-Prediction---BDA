import React from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate, Link } from "react-router-dom";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Button } from "./ui/button";
import { Heart, AlertCircle, LogOut } from "lucide-react";
import authService from "../services/authService";
import apiService from "../services/api";

const UserProfile = () => {
  const navigate = useNavigate();
  const currentUser = authService.getCurrentUser();

  const {
    data: predictions,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["userPredictions"],
    queryFn: apiService.getUserPredictions,
    enabled: !!currentUser,
  });

  const handleLogout = () => {
    authService.logout();
    navigate("/login");
  };

  if (!currentUser) {
    navigate("/login");
    return null;
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">My Profile</h1>
          <p className="text-muted-foreground">
            Welcome back, {currentUser.username}!
          </p>
        </div>
        <Button variant="outline" onClick={handleLogout}>
          <LogOut className="mr-2 w-4 h-4" />
          Logout
        </Button>
      </div>

      <div className="grid gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Heart className="mr-2 w-5 h-5 text-red-500" />
              My Predictions
            </CardTitle>
            <CardDescription>
              View your saved heart health predictions
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <p className="py-4 text-center">Loading your predictions...</p>
            ) : error ? (
              <div className="flex gap-3 items-start p-4 text-red-800 bg-red-50 rounded-md border border-red-200">
                <AlertCircle className="mt-0.5 w-5 h-5" />
                <div>
                  <p>Failed to load predictions</p>
                  <p className="text-sm">
                    {error.message || "Please try again later"}
                  </p>
                </div>
              </div>
            ) : predictions?.length === 0 ? (
              <div className="py-8 text-center">
                <p className="mb-4 text-muted-foreground">
                  You haven't made any predictions yet
                </p>
                <Link to="/predict">
                  <Button>Make Your First Prediction</Button>
                </Link>
              </div>
            ) : (
              <div className="space-y-4">
                {predictions?.map((prediction) => (
                  <Card key={prediction.id} className="overflow-hidden">
                    <div
                      className={`h-2 ${
                        prediction.prediction_result.cardio === 1
                          ? "bg-red-500"
                          : "bg-green-500"
                      }`}
                    ></div>
                    <CardContent className="p-4">
                      <div className="flex justify-between items-start">
                        <div>
                          <h4 className="font-medium">
                            {new Date(
                              prediction.created_at
                            ).toLocaleDateString()}
                          </h4>
                          <p className="text-sm text-muted-foreground">
                            Risk Level:{" "}
                            {prediction.prediction_result.risk_level}
                          </p>
                        </div>
                        <div
                          className={`px-3 py-1 rounded-full text-sm font-medium ${
                            prediction.prediction_result.cardio === 1
                              ? "bg-red-100 text-red-800"
                              : "bg-green-100 text-green-800"
                          }`}
                        >
                          {prediction.prediction_result.cardio === 1
                            ? "Risk Detected"
                            : "No Risk"}
                        </div>
                      </div>
                    </CardContent>
                    <CardFooter className="flex justify-end p-4 bg-muted/50">
                      <Link to={`/results/${prediction.id}`}>
                        <Button variant="outline" size="sm">
                          View Details
                        </Button>
                      </Link>
                    </CardFooter>
                  </Card>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default UserProfile;
