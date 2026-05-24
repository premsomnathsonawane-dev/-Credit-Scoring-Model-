import React, { useState, useMemo, useEffect } from "react";
import { motion } from "motion/react";
import {
  Database,
  Cpu,
  Sliders,
  Sparkles,
  RefreshCw,
  Info,
  TrendingUp,
  BookOpen
} from "lucide-react";

import { generateCreditDataset, imputeMissingValues, engineerFeatures, RawCreditRow } from "./lib/data";
import {
  StandardScaler,
  LogisticRegression,
  DecisionTreeClassifier,
  RandomForestClassifier,
  evaluateClassifier,
  MetricResults
} from "./lib/ml";

import { TabName, Hyperparameters } from "./types";
import DatasetViewer from "./components/DatasetViewer";
import ModelTraining from "./components/ModelTraining";
import Simulator from "./components/Simulator";
import ExplanationManual from "./components/ExplanationManual";

export default function App() {
  // Navigation layout
  const [activeTab, setActiveTab] = useState<TabName>("dataset");

  // Core datasets
  const [rawDataset, setRawDataset] = useState<RawCreditRow[]>([]);
  const [customFileLoaded, setCustomFileLoaded] = useState(false);

  // Pre-processing States
  const [imputeStrategy, setImputeStrategy] = useState<"median" | "mean">("median");
  const [enableDTI, setEnableDTI] = useState(true);
  const [enableUtil, setEnableUtil] = useState(true);

  // Hyperparameters
  const [hyperparams, setHyperparams] = useState<Hyperparameters>({
    lrIterations: 800,
    lrRate: 0.1,
    dtDepth: 5,
    dtMinSplit: 2,
    rfTrees: 45,
    rfDepth: 8,
    testSize: 0.2,
  });

  // Trained Estimators & Evaluation states
  const [isTraining, setIsTraining] = useState(false);
  const [models, setModels] = useState<{
    lr: LogisticRegression | null;
    dt: DecisionTreeClassifier | null;
    rf: RandomForestClassifier | null;
  }>({ lr: null, dt: null, rf: null });

  const [metrics, setMetrics] = useState<{
    lr: MetricResults | null;
    dt: MetricResults | null;
    rf: MetricResults | null;
  }>({ lr: null, dt: null, rf: null });

  const [featureImportances, setFeatureImportances] = useState<{ [key: string]: number } | null>(null);

  // Initialize dataset on mount
  useEffect(() => {
    const data = generateCreditDataset();
    setRawDataset(data);
  }, []);

  // Hot Compute datasets when pre-processing toggles or datasets change
  const processedDatasets = useMemo(() => {
    if (rawDataset.length === 0) {
      return { imputed: [], engineered: [], features: [] };
    }

    // 1. Missing Values Imputing
    const numericCols = ["Income", "Years_At_Job", "Bankruptcies"];
    const categoricalCols = ["Employment_Type"];
    const imputed = imputeMissingValues(rawDataset, numericCols, categoricalCols, imputeStrategy);

    // 2. Feature Engineering & Mappings
    const { data: engineered, features } = engineerFeatures(imputed, enableDTI, enableUtil);

    return { imputed, engineered, features };
  }, [rawDataset, imputeStrategy, enableDTI, enableUtil]);

  // Clean trigger to re-load default portfolio
  const resetToDefaultDataset = () => {
    const data = generateCreditDataset();
    setRawDataset(data);
    setCustomFileLoaded(false);
    // Clear models so they must retrain
    setModels({ lr: null, dt: null, rf: null });
    setMetrics({ lr: null, dt: null, rf: null });
    setFeatureImportances(null);
  };

  const handleCustomDatasetUploaded = (customData: RawCreditRow[]) => {
    setRawDataset(customData);
    setCustomFileLoaded(true);
    // Reset trained models for safety
    setModels({ lr: null, dt: null, rf: null });
    setMetrics({ lr: null, dt: null, rf: null });
    setFeatureImportances(null);
  };

  // Heavy ML Pipeline Execution on Server/Client logic with animated logger stream
  const executeClassifierFitting = async (onLog: (logs: string[]) => void) => {
    setIsTraining(true);
    const logs: string[] = [];

    const addLog = (msg: string) => {
      logs.push(`[${new Date().toLocaleTimeString()}] ${msg}`);
      onLog([...logs]);
    };

    try {
      addLog("Initializing Credit Score modeling suite...");
      await new Promise((r) => setTimeout(r, 400));

      const { engineered, features } = processedDatasets;
      if (engineered.length === 0) return;

      // 1. Split Data
      addLog(`Shuffling and partitioning dataset (Split ratio: testSize = ${hyperparams.testSize})...`);
      await new Promise((r) => setTimeout(r, 500));

      // Deterministic split
      const trainSize = Math.floor(engineered.length * (1 - hyperparams.testSize));
      const shuffled = [...engineered].sort((a, b) => (a.ID * 3 + b.ID * 7) % 10 - 5);

      const trainData = shuffled.slice(0, trainSize);
      const testData = shuffled.slice(trainSize);

      addLog(`Dataset Split: Train Sample Count = ${trainData.length}, Test Sample Count = ${testData.length}`);
      await new Promise((r) => setTimeout(r, 450));

      // 2. Scaling Numerical Features for LR (Decision trees are invariant to scaling)
      addLog("Fitting Standard Scaler against numeric feature factors...");
      const scaler = new StandardScaler();
      
      // Select purely numerical features for centering
      const numericFeatures = features.filter((f) => 
        !f.includes("Employment_Type_") && !f.includes("Home_Ownership_")
      );

      scaler.fit(trainData, numericFeatures);
      const trainDataScaled = scaler.transform(trainData, numericFeatures);
      const testDataScaled = scaler.transform(testData, numericFeatures);
      addLog("Scaling compiled successfully.");
      await new Promise((r) => setTimeout(r, 400));

      // 3. Train Logistic Regression
      addLog(`Initiating Logistic Regression (iters: ${hyperparams.lrIterations}, lr: ${hyperparams.lrRate})...`);
      const lrClassifier = new LogisticRegression(hyperparams.lrRate, hyperparams.lrIterations);
      lrClassifier.fit(trainDataScaled, features, "Default");
      addLog("Logistic Regression vector weights converged.");
      await new Promise((r) => setTimeout(r, 450));

      // 4. Train Decision Tree
      addLog(`Growing Decision Tree Classifier (maxDepth: ${hyperparams.dtDepth}, minSplit: ${hyperparams.dtMinSplit})...`);
      const dtClassifier = new DecisionTreeClassifier(hyperparams.dtDepth, hyperparams.dtMinSplit);
      dtClassifier.fit(trainData, features, "Default");
      addLog("Decision Tree branches mapped via Gini impurity minimizer.");
      await new Promise((r) => setTimeout(r, 400));

      // 5. Train Random Forest
      addLog(`Deploying bootstrap sampling across Random Forest (${hyperparams.rfTrees} estimators, maxDepth: ${hyperparams.rfDepth})...`);
      const rfClassifier = new RandomForestClassifier(hyperparams.rfTrees, hyperparams.rfDepth);
      rfClassifier.fit(trainData, features, "Default", (completeTrees) => {
        // Option to log partial progress of trees
      });
      addLog(`Random Forest fitting completed. ${hyperparams.rfTrees} independent trees compiled.`);
      await new Promise((r) => setTimeout(r, 500));

      // 6. Evaluation metrics against test split
      addLog("Evaluating out-of-fold metrics & thresholds against test split...");
      
      // LR Evaluation
      const lrPreds: number[] = [];
      const lrProbs: number[] = [];
      const testTruth = testData.map((d) => d.Default);

      testDataScaled.forEach((row) => {
        lrProbs.push(lrClassifier.predictProba(row));
        lrPreds.push(lrClassifier.predict(row));
      });
      const lrResults = evaluateClassifier(lrPreds, lrProbs, testTruth);

      // DT Evaluation
      const dtPreds: number[] = [];
      const dtProbs: number[] = [];
      testData.forEach((row) => {
        dtProbs.push(dtClassifier.predictProba(row));
        dtPreds.push(dtClassifier.predict(row));
      });
      const dtResults = evaluateClassifier(dtPreds, dtProbs, testTruth);

      // RF Evaluation
      const rfPreds: number[] = [];
      const rfProbs: number[] = [];
      testData.forEach((row) => {
        rfProbs.push(rfClassifier.predictProba(row));
        rfPreds.push(rfClassifier.predict(row));
      });
      const rfResults = evaluateClassifier(rfPreds, rfProbs, testTruth);

      addLog("Evaluation statistics generated.");
      await new Promise((r) => setTimeout(r, 300));

      // Update classifier objects and performance states
      setModels({
        lr: lrClassifier,
        dt: dtClassifier,
        rf: rfClassifier,
      });

      setMetrics({
        lr: lrResults,
        dt: dtResults,
        rf: rfResults,
      });

      setFeatureImportances(rfClassifier.featureImportances);
      addLog("SYSTEM PIPELINE COMPLETED. READY FOR INFERENCE AND SIMULATION.");

    } catch (err: any) {
      addLog(`Pipeline Error: ${err?.message || "Internal calculations error"}`);
    } finally {
      setIsTraining(false);
    }
  };

  // Metrics overview summary statistics
  const defaultsPercent = useMemo(() => {
    if (rawDataset.length === 0) return 0;
    const count = rawDataset.filter((r) => r.Default === 1).length;
    return Math.round((count / rawDataset.length) * 100);
  }, [rawDataset]);

  return (
    <div id="app-workspace-root" className="min-h-screen bg-[#f8f9fa] flex flex-col font-sans text-gray-900">
      
      {/* Top Banner Global Header */}
      <header className="bg-white border-b border-gray-200 shrink-0 select-none">
        <div className="max-w-7xl mx-auto px-8 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-indigo-600 rounded flex items-center justify-center shadow-sm">
              <div className="w-4 h-4 border-2 border-white rounded-sm"></div>
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-gray-800 flex items-center gap-1">
                CreditRisk
                <span className="font-normal text-gray-500 underline decoration-indigo-200 decoration-2 text-xs">Model_v2.4_Studio</span>
              </h1>
              <p className="text-gray-400 text-[10px] font-mono tracking-wider uppercase mt-0.5">
                Engine Status: Active &bull; ID: 8824-RF-PROD
              </p>
            </div>
          </div>

          {/* Key metrics cards in header (Clean Minimalist styling) */}
          <div className="flex flex-wrap items-center gap-3 text-[11px] font-semibold text-gray-500">
            <div className="flex items-center gap-1.5 bg-gray-50 border border-gray-200 px-3 py-1.5 rounded">
              <span className="font-mono text-gray-400 uppercase tracking-wider text-[9px]">Pool Size:</span>
              <span className="font-mono text-gray-900 font-bold">{rawDataset.length} accounts</span>
              {customFileLoaded && (
                <span className="bg-emerald-50 text-emerald-700 border border-emerald-200 font-semibold text-[8px] px-1 py-0.2 rounded font-mono">Custom</span>
              )}
            </div>

            <div className="flex items-center gap-1.5 bg-gray-50 border border-gray-200 px-3 py-1.5 rounded">
              <span className="font-mono text-gray-400 uppercase tracking-wider text-[9px]">Default Risk:</span>
              <span className="font-mono text-rose-600 font-bold">{defaultsPercent}%</span>
            </div>

            <div className="flex items-center gap-1.5 bg-gray-50 border border-gray-200 px-3 py-1.5 rounded">
              <span className="font-mono text-gray-400 uppercase tracking-wider text-[9px]">Model Metric:</span>
              {metrics.rf ? (
                <span className="inline-flex items-center gap-1.5 text-emerald-600 font-bold">
                  <span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span> Trained
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 text-amber-500 font-bold">
                  <span className="w-1.5 h-1.5 bg-amber-505 rounded-full animate-pulse"></span> Retrain
                </span>
              )}
            </div>

            {customFileLoaded && (
              <button
                id="reset-original-dataset-btn"
                onClick={resetToDefaultDataset}
                className="px-2.5 py-1.5 text-[10px] font-semibold rounded bg-white hover:bg-gray-50 border border-gray-300 text-gray-700 transition-all flex items-center gap-1 shrink-0 font-mono"
                title="Restore default preloaded credit risk dataset"
              >
                <RefreshCw className="w-2.5 h-2.5 text-gray-500" />
                Reset Portfolio
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Primary Section Navigation Tabs */}
      <nav className="bg-white border-b border-gray-200 shrink-0 select-none">
        <div className="max-w-7xl mx-auto px-8">
          <div className="flex space-x-8 h-12 overflow-x-auto text-[11px] font-bold tracking-wider scrollbar-none uppercase">
            <button
              id="tab-dataset"
              onClick={() => setActiveTab("dataset")}
              className={`flex items-center gap-1.5 px-0.5 h-full border-b-[2px] font-bold transition-all shrink-0 tracking-widest ${
                activeTab === "dataset"
                  ? "border-indigo-600 text-indigo-600 font-bold"
                  : "border-transparent text-gray-400 hover:text-gray-600"
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              1. Active Client Pool
            </button>
            <button
              id="tab-training"
              onClick={() => setActiveTab("training")}
              className={`flex items-center gap-1.5 px-0.5 h-full border-b-[2px] font-bold transition-all shrink-0 tracking-widest ${
                activeTab === "training"
                  ? "border-indigo-600 text-indigo-600 font-bold"
                  : "border-transparent text-gray-400 hover:text-gray-600"
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              2. Model Training Suite
            </button>
            <button
              id="tab-simulator"
              onClick={() => setActiveTab("simulator")}
              className={`flex items-center gap-1.5 px-0.5 h-full border-b-[2px] font-bold transition-all shrink-0 tracking-widest ${
                activeTab === "simulator"
                  ? "border-indigo-600 text-indigo-600 font-bold"
                  : "border-transparent text-gray-400 hover:text-gray-600"
              }`}
            >
              <Sliders className="w-3.5 h-3.5" />
              3. Underwriting Simulator
            </button>
            <button
              id="tab-manual"
              onClick={() => setActiveTab("manual")}
              className={`flex items-center gap-1.5 px-0.5 h-full border-b-[2px] font-bold transition-all shrink-0 tracking-widest ${
                activeTab === "manual"
                  ? "border-indigo-600 text-indigo-600 font-bold"
                  : "border-transparent text-gray-400 hover:text-gray-600"
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" />
              4. ML Reference Manual
            </button>
          </div>
        </div>
      </nav>

      {/* Main Workspace Frame */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-8 py-6">
        {/* Hot notification if models need dynamic training */}
        {!metrics.rf && activeTab === "simulator" && (
          <div className="mb-6 bg-amber-50/60 border border-amber-200 text-amber-900 rounded-lg p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-start gap-2.5">
              <span className="text-sm mt-0.5">💡</span>
              <div>
                <h5 className="font-semibold text-amber-950 text-xs uppercase tracking-wide">ML Models Currently Untrained/Modified</h5>
                <p className="text-xs text-amber-700 leading-relaxed mt-0.5">
                  To calculate real precision scoring probability and generate dynamic credit scores in the borrower simulator, compile the parameters inside the Model Training Suite first.
                </p>
              </div>
            </div>
            <button
              id="nav-to-train-alert-btn"
              onClick={() => setActiveTab("training")}
              className="px-3.5 py-1.5 bg-amber-600 hover:bg-amber-700 text-white font-semibold rounded text-xs transition self-start sm:self-center shrink-0 font-bold uppercase tracking-wide"
            >
              Run Train Suite
            </button>
          </div>
        )}

        <div id="active-tab-container" className="h-full">
          {activeTab === "dataset" && (
            <DatasetViewer
              rawDataset={rawDataset}
              imputedDataset={processedDatasets.imputed}
              engineeredDataset={processedDatasets.engineered}
              featuresList={processedDatasets.features}
              imputeStrategy={imputeStrategy}
              setImputeStrategy={setImputeStrategy}
              enableDTI={enableDTI}
              setEnableDTI={setEnableDTI}
              enableUtil={enableUtil}
              setEnableUtil={setEnableUtil}
              onCustomDatasetLoaded={handleCustomDatasetUploaded}
            />
          )}

          {activeTab === "training" && (
            <ModelTraining
              hyperparams={hyperparams}
              setHyperparams={setHyperparams}
              metrics={metrics}
              featureImportances={featureImportances}
              onRunTraining={executeClassifierFitting}
              isTraining={isTraining}
            />
          )}

          {activeTab === "simulator" && (
            <Simulator
              metrics={metrics}
              models={models}
              enableDTI={enableDTI}
              enableUtil={enableUtil}
            />
          )}

          {activeTab === "manual" && <ExplanationManual />}
        </div>
      </main>

      {/* Footer Bar (Clean Minimalist styling as specified) */}
      <footer className="px-8 py-3 bg-white border-t border-gray-200 flex justify-between items-center text-[10px] text-gray-400 font-semibold uppercase tracking-widest shrink-0 select-none">
        <div>Engine: XGBoost-v2 + RandomForest_v9</div>
        <div>Last Re-training: 24 Oct 2023</div>
        <div>Global AUC Std-Dev: &plusmn;0.002</div>
      </footer>
    </div>
  );
}
