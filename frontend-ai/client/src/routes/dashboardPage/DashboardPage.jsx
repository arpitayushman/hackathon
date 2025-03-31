import { useLocation } from "react-router-dom";
import { useState,useEffect } from "react";
import "./dashboardPage.css";
import { Line } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from "chart.js";
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const DashboardPage = () => {
  const location = useLocation();
  const { csvData, imageUrls } = location.state || {}; // Get passed data
  console.log("CSV Data:", csvData);
  console.log("Image URLs:", imageUrls);
  const [selectedYear, setSelectedYear] = useState("");
  const [filteredData, setFilteredData] = useState(csvData);
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 4;

  // Compute total pages
  const totalPages = csvData ? Math.ceil(csvData.length / rowsPerPage) : 1;
  const years = [...new Set(csvData ? csvData.map((row) => new Date(row.date).getFullYear()) : [])];
  useEffect(() => {
    if (selectedYear) {
      setFilteredData(csvData.filter((row) => new Date(row.date).getFullYear() === parseInt(selectedYear)));
    } else {
      setFilteredData(csvData); // Show all data if no year is selected
    }
  }, [selectedYear, csvData]);
  // Get data for the current page
  const paginatedData = csvData
    ? csvData.slice((currentPage - 1) * rowsPerPage, currentPage * rowsPerPage)
    : [];
    const chartData = {
      labels: filteredData  ? filteredData .map((row) => row["date"]) : [], // Assuming 'timestamp' is the time or x-axis data
      datasets: [
        {
          label: "Predicted Average Amount", 
          data: filteredData  ? filteredData .map((row) => row["predicted_avg_amount"]) : [], 
          fill: false,
          borderColor: "rgba(75,192,192,1)",
          tension: 0.1,
        },
        {
          label: "predicted_total_amount", // Dataset name
          data: filteredData  ? filteredData .map((row) => row["predicted_total_amount"]) : [], 
          fill: false,
          borderColor: "rgba(255,99,132,1)",
          tension: 0.1,
        },
        {
          label: "predicted_txn_count", // Dataset name
          data: filteredData  ? filteredData .map((row) => row["predicted_txn_count"]) : [], 
          fill: false,
          borderColor: "rgb(255, 255, 99)",
          tension: 0.1,
        },
      ],
    };
    const options = {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: 'Prediction Data Over Time',
        },
        tooltip: {
          mode: 'index',
          intersect: false,
        },
      },
      scales: {
        x: {
          type: 'category',
          title: {
            display: true,
            text: 'Date',
          },
        },
        y: {
          type: 'linear',
          title: {
            display: true,
            text: 'Amount/Transaction Count',
          },
          ticks: {
            beginAtZero: true,
          },
        },
      },
    };
    const convertToCSV = (data) => {
      const header = Object.keys(data[0]).join(","); // Get headers
      const rows = data.map(row =>
        Object.values(row).map(value => `"${value}"`).join(",") // Wrap values in quotes
      );
      return [header, ...rows].join("\n");
    };
    const downloadCSV = () => {
      if (!filteredData || filteredData.length === 0) return;
      const csv = convertToCSV(filteredData);
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const link = document.createElement("a");
      if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", "prediction_data.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    };
    
  return (
    <div className="dashboardPage">
      {/* <div className="texts">
        <div className="logo">
          <img src="/logo.png" alt="Logo" />
          <div style={{ display: "flex", flexDirection: "column" }}>
            <h2>RunTime Terror AI Predictive Analytics</h2>
          </div>
        </div>
      </div> */}
      {/* <div className="images-container">
        {imageUrls && Object.keys(imageUrls).length > 0 ? (
          <>
           
            {Object.entries(imageUrls).map(([fileName, url]) => (
              <div key={fileName} className="image-item">
                <p className="filename">{fileName.replace(".png","")}</p>
                <img src={url} alt={fileName} />
              </div>
            ))}
          </>
        ) : (
          <p>No images available.</p>
        )}
      </div> */}
      {csvData && csvData.length > 0 ? (
        <div className="chart-container">
          <h3>Predective Analysis</h3>
          <select value={selectedYear} onChange={(e) => setSelectedYear(e.target.value)} className="year-selector">
            <option value="">Select Year</option>
            {years.map((year) => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>
          <Line data={chartData} options={options}/>
        </div>
      ) : (
        <p>No data available for the plot.</p>
      )}

      {/* 📝 CSV Table */}
      {csvData && csvData.length > 0 ? (
        <div className="table-container">
        
          <h2 style={{marginBottom:"15px",fontSize:"25px"}}> Prediction Data</h2>
        <div style={{display:"flex",width:"100%",justifyContent:"flex-end"}}> <button className="download-csv-btn" onClick={downloadCSV}>
            Download CSV
          </button></div>
          <table className="tableclass">
            <thead className="tablehead">
              <tr>
                {Object.keys(csvData[0]).map((header, index) => (
                  <th key={index}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {paginatedData.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {Object.values(row).map((cell, cellIndex) => (
                    <td key={cellIndex}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>

          {/* Pagination Controls */}
          <div className="pagination">
            <button 
              onClick={() => setCurrentPage(currentPage - 1)} 
              disabled={currentPage === 1}
            >
              Prev
            </button>
            <span>Page {currentPage} of {totalPages}</span>
            <button 
              onClick={() => setCurrentPage(currentPage + 1)} 
              disabled={currentPage === totalPages}
            >
              Next
            </button>
          </div>
         
        </div>
      ) : (
        <p>No CSV data available.</p>
      )}


    </div>


  );

};

export default DashboardPage;
