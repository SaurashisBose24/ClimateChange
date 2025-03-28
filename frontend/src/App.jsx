import React, { useEffect, useState } from "react";
import "./App.css";
import { Line } from "react-chartjs-2";
import axios from "axios";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const ClimateTrendChart = () => {
  const [chartData, setChartData] = useState(null);
  const [historicData, setHistoricData] = useState(null);

  useEffect(() => {
    // Fetch future predictions
    axios.get("http://127.0.0.1:5000/predict")
      .then((response) => {
        const { years, predictions } = response.data;
        setChartData({
          labels: years,
          datasets: [
            {
              label: "Predicted Temperature Anomaly (°C)",
              data: predictions,
              borderColor: "red",
              backgroundColor: "rgba(255, 0, 0, 0.5)",
              tension: 0.4,
              fill: true,
              borderWidth: 2,
              shadowOffsetX: 4,
              shadowOffsetY: 4,
              shadowBlur: 15,
              shadowColor: "rgba(255, 0, 0, 0.5)",
            },
          ],
        });
      })
      .catch((error) => console.error("Error fetching prediction data:", error));

    // Fetch historical data
    axios.get("http://127.0.0.1:5000/historic")
      .then((response) => {
        const { years, anomalies } = response.data;
        setHistoricData({
          labels: years,
          datasets: [
            {
              label: "Historical Temperature Anomaly (°C)",
              data: anomalies,
              borderColor: "blue",
              backgroundColor: "rgba(0, 0, 255, 0.5)",
              tension: 0.4,
              fill: true,
              borderWidth: 2,
              shadowOffsetX: 4,
              shadowOffsetY: 4,
              shadowBlur: 15,
              shadowColor: "rgba(0, 0, 0, 0.5)",
            },
          ],
        });
      })
      .catch((error) => console.error("Error fetching historic data:", error));
  }, []);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 2000, // Animation duration in milliseconds
      easing: 'easeInOutQuart', // Smooth animation effect
    },
  };

  return (
    <div className="M">
      <h2>Climate Trend Analysis</h2>
      <div style={{ width: '90vw', height: '300px' }}>
        {historicData && <Line data={historicData} options={chartOptions} />}
      </div>
      <div style={{ width: '90vw', height: '300px' }}>
        {chartData && <Line data={chartData} options={chartOptions}/>}
      </div>
      {(!historicData || !chartData) && <p>Loading...</p>}
    </div>
  );
};

export default ClimateTrendChart;
