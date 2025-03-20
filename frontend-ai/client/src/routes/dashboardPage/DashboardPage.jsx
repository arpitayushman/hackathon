import { useLocation } from "react-router-dom";
import { useState } from "react";
import "./dashboardPage.css";

const DashboardPage = () => {
  const location = useLocation();
  const { csvData, imageUrls } = location.state || {}; // Get passed data
  console.log("CSV Data:", csvData);
  console.log("Image URLs:", imageUrls);
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 4;

  // Compute total pages
  const totalPages = csvData ? Math.ceil(csvData.length / rowsPerPage) : 1;

  // Get data for the current page
  const paginatedData = csvData
    ? csvData.slice((currentPage - 1) * rowsPerPage, currentPage * rowsPerPage)
    : [];

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
      <div className="images-container">
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
      </div>

      {/* 📝 CSV Table */}
      {csvData && csvData.length > 0 ? (
        <div className="table-container">
        
          <h2 style={{marginBottom:"15px",fontSize:"25px"}}> Prediction Data</h2>
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
