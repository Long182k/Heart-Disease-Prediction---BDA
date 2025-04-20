import React, { useState, useEffect } from "react";
import { AlertCircle } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const ModelMetrics = () => {
  // Use local state instead of React Query
  const [metrics, setMetrics] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load metrics from the JSON file
  useEffect(() => {
    // Hardcoded metrics data from model_metrics.json
    const metricsData = [
      {
        model_name: "logistic_regression",
        accuracy: 0.7268712899956565,
        precision: 0.7544657356284508,
        recall: 0.6726509338352397,
        f1: 0.7112131649445083,
        auc: 0.7883863106796707,
        training_time: 73.04523634910583,
        best_params: {
          C: 1,
          class_weight: null,
          dual: false,
          fit_intercept: true,
          intercept_scaling: 1,
          l1_ratio: null,
          max_iter: 1000,
          multi_class: "deprecated",
          n_jobs: null,
          penalty: "l2",
          random_state: 42,
          solver: "saga",
          tol: 0.0001,
          verbose: 0,
          warm_start: false
        },
        optimal_threshold: 0.48301090477114256
      },
      {
        model_name: "random_forest",
        accuracy: 0.7312871000434342,
        precision: 0.751060820367751,
        recall: 0.6919067612566961,
        f1: 0.7202712886209495,
        auc: 0.7987660692747461,
        training_time: 347.08749866485596,
        best_params: {
          bootstrap: true,
          ccp_alpha: 0.0,
          class_weight: null,
          criterion: "gini",
          max_depth: 10,
          max_features: "sqrt",
          max_leaf_nodes: null,
          max_samples: null,
          min_impurity_decrease: 0.0,
          min_samples_leaf: 1,
          min_samples_split: 5,
          min_weight_fraction_leaf: 0.0,
          monotonic_cst: null,
          n_estimators: 200,
          n_jobs: null,
          oob_score: false,
          random_state: 42,
          verbose: 0,
          warm_start: false
        },
        optimal_threshold: 0.5246942451857808
      },
      {
        model_name: "gradient_boosting",
        accuracy: 0.7311423193861301,
        precision: 0.7504313725490196,
        recall: 0.692630664543217,
        f1: 0.7203734377352808,
        auc: 0.7998448916418338,
        training_time: 126.29886245727539,
        best_params: {
          ccp_alpha: 0.0,
          criterion: "friedman_mse",
          init: null,
          learning_rate: 0.1,
          loss: "log_loss",
          max_depth: 5,
          max_features: null,
          max_leaf_nodes: null,
          min_impurity_decrease: 0.0,
          min_samples_leaf: 1,
          min_samples_split: 2,
          min_weight_fraction_leaf: 0.0,
          n_estimators: 100,
          n_iter_no_change: null,
          random_state: 42,
          subsample: 1.0,
          tol: 0.0001,
          validation_fraction: 0.1,
          verbose: 0,
          warm_start: false
        },
        optimal_threshold: 0.4646813418363161
      },
      {
        model_name: "xgboost",
        accuracy: 0.7315766613580426,
        precision: 0.7521677439697304,
        recall: 0.6907485159982626,
        f1: 0.7201509433962264,
        auc: 0.7997073217194526,
        training_time: 7.363225221633911,
        best_params: {
          objective: "binary:logistic",
          base_score: null,
          booster: null,
          callbacks: null,
          colsample_bylevel: null,
          colsample_bynode: null,
          colsample_bytree: null,
          device: null,
          early_stopping_rounds: null,
          enable_categorical: false,
          eval_metric: null,
          feature_types: null,
          gamma: null,
          grow_policy: null,
          importance_type: null,
          interaction_constraints: null,
          learning_rate: 0.1,
          max_bin: null,
          max_cat_threshold: null,
          max_cat_to_onehot: null,
          max_delta_step: null,
          max_depth: 3,
          max_leaves: null,
          min_child_weight: null,
          missing: NaN,
          monotone_constraints: null,
          multi_strategy: null,
          n_estimators: 100,
          n_jobs: null,
          num_parallel_tree: null,
          random_state: 42,
          reg_alpha: null,
          reg_lambda: null,
          sampling_method: null,
          scale_pos_weight: null,
          subsample: null,
          tree_method: null,
          validate_parameters: null,
          verbosity: null
        },
        optimal_threshold: 0.5259686708450317
      },
      {
        model_name: "ensemble",
        accuracy: 0.7340379325322137,
        precision: 0.7561400728886072,
        recall: 0.6908932966555669,
        f1: 0.7220456952640338,
        auc: 0.7992316752325237,
        training_time: 24.379757404327393,
        optimal_threshold: 0.5005820006603595
      },
    ];

    try {
      setMetrics(metricsData);
      setIsLoading(false);
    } catch (err) {
      console.error("Error loading metrics data:", err);
      setError(err);
      setIsLoading(false);
    }
  }, []);

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "Model Performance Comparison",
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
      },
    },
  };

  // Time chart options
  const timeOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "Model Training Times (seconds)",
      },
    },
  };

  // Format model name for display
  const formatModelName = (name) => {
    return name
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  // Prepare performance chart data
  const prepareChartData = () => {
    if (!Array.isArray(metrics)) return { labels: [], datasets: [] };

    const labels = metrics.map((m) => formatModelName(m.model_name));
    console.log("🚀 labels:", labels);

    return {
      labels,
      datasets: [
        {
          label: "Accuracy",
          data: metrics.map((m) => m.accuracy),
          backgroundColor: "rgba(54, 162, 235, 0.6)",
          borderColor: "rgba(54, 162, 235, 1)",
          borderWidth: 1,
        },
        {
          label: "AUC",
          data: metrics.map((m) => m.auc),
          backgroundColor: "rgba(153, 102, 255, 0.6)",
          borderColor: "rgba(153, 102, 255, 1)",
          borderWidth: 1,
        },
        {
          label: "F1 Score",
          data: metrics.map((m) => m.f1),
          backgroundColor: "rgba(255, 159, 64, 0.6)",
          borderColor: "rgba(255, 159, 64, 1)",
          borderWidth: 1,
        },
      ],
    };
  };

  // Prepare time chart data
  const prepareTimeChartData = () => {
    if (!Array.isArray(metrics)) return { labels: [], datasets: [] };

    const labels = metrics.map((m) => formatModelName(m.model_name));

    return {
      labels,
      datasets: [
        {
          label: "Training Time (seconds)",
          data: metrics.map((m) => m.training_time),
          backgroundColor: "rgba(255, 99, 132, 0.6)",
          borderColor: "rgba(255, 99, 132, 1)",
          borderWidth: 1,
        },
      ],
    };
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="w-12 h-12 rounded-full border-b-2 animate-spin border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 rounded-lg border-destructive/50 bg-destructive/5">
        <div className="flex gap-2 items-center">
          <AlertCircle className="w-5 h-5 text-destructive" />
          <h3 className="font-medium">Failed to load model metrics</h3>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          Please refresh the page or try again later.
        </p>
      </div>
    );
  }

  // if (!metrics || !Array.isArray(metrics) || metrics.length === 0) {
  //   return (
  //     <div className="p-8 text-center">
  //       <h2 className="text-xl font-semibold">No model metrics available</h2>
  //       <p className="mt-2 text-muted-foreground">
  //         Model performance data has not been generated yet.
  //       </p>
  //     </div>
  //   );
  // }

  // Find best model based on AUC score
  const bestModel = Array.isArray(metrics)
    ? metrics.reduce(
        (prev, current) => (prev.auc > current.auc ? prev : current),
        metrics[0]
      )
    : null;

  return (
    <div className="space-y-8">
      <div className="space-y-2 text-center">
        <h1 className="text-3xl font-bold">Model Performance Metrics</h1>
        <p className="text-muted-foreground">
          Comparing the performance of different machine learning models for
          heart disease prediction
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Performance Metrics</CardTitle>
            <CardDescription>
              Comparing accuracy, AUC, and F1 scores across models
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Bar data={prepareChartData()} options={options} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Training Time</CardTitle>
            <CardDescription>
              Time taken to train each model (in seconds)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Bar data={prepareTimeChartData()} options={timeOptions} />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Detailed Model Metrics</CardTitle>
          <CardDescription>
            Comprehensive performance metrics for each model
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b">
                  <th className="px-4 py-3 font-medium text-left">Model</th>
                  <th className="px-4 py-3 font-medium text-left">Accuracy</th>
                  <th className="px-4 py-3 font-medium text-left">Precision</th>
                  <th className="px-4 py-3 font-medium text-left">Recall</th>
                  <th className="px-4 py-3 font-medium text-left">F1 Score</th>
                  <th className="px-4 py-3 font-medium text-left">AUC</th>
                  <th className="px-4 py-3 font-medium text-left">Threshold</th>
                  <th className="px-4 py-3 font-medium text-left">
                    Training Time (s)
                  </th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((model) => (
                  <tr
                    key={model.model_name}
                    className={`border-b hover:bg-muted/50 ${
                      model.model_name === bestModel.model_name
                        ? "bg-primary/5"
                        : ""
                    }`}
                  >
                    <td className="px-4 py-3 font-medium">
                      {formatModelName(model.model_name)}
                    </td>
                    <td className="px-4 py-3">{model.accuracy?.toFixed(4)}</td>
                    <td className="px-4 py-3">{model.precision?.toFixed(4)}</td>
                    <td className="px-4 py-3">{model.recall?.toFixed(4)}</td>
                    <td className="px-4 py-3">{model.f1?.toFixed(4)}</td>
                    <td className="px-4 py-3">{model.auc?.toFixed(4)}</td>
                    <td className="px-4 py-3">
                      {model.optimal_threshold?.toFixed(4)}
                    </td>
                    <td className="px-4 py-3">
                      {model.training_time?.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-primary/5 border-primary/20">
        <CardHeader>
          <CardTitle>Best Performing Model</CardTitle>
        </CardHeader>
        <CardContent>
          <p>
            The{" "}
            <span className="font-bold">
              {formatModelName(bestModel.model_name)}
            </span>{" "}
            model achieved the highest AUC score of{" "}
            <span className="font-bold">{bestModel.auc?.toFixed(4)}</span> with
            an optimal threshold of{" "}
            <span className="font-bold">
              {bestModel.optimal_threshold?.toFixed(4)}
            </span>
            .
          </p>
          <p className="mt-2 text-sm text-muted-foreground">
            AUC (Area Under the ROC Curve) is a key metric for evaluating
            classification model performance, especially for imbalanced datasets
            like those commonly found in medical applications.
          </p>

          {bestModel.best_params && (
            <div className="mt-4">
              <h3 className="mb-2 text-sm font-medium">Best Parameters:</h3>
              <div className="overflow-auto p-3 max-h-40 text-xs rounded-md bg-muted/50">
                <pre>{JSON.stringify(bestModel.best_params, null, 2)}</pre>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelMetrics;
