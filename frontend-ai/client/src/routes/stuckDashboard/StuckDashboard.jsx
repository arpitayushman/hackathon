import { useLocation } from "react-router-dom";
import { useState } from "react";
// import "./dashboardPage.css";
import "./stuckdashboard.css";

const StuckDashboard = () => {
const stuckData=[

    {
    "refid": "MUTH00000NATYF400063ST01HINOSHUOIN",
    "last_upt_ts": "2025-03-25 12:48:07.240",
    "duration_in_minutes" :20,
    "leg" :"Request send to BOU"
  
    
    },
    {
    "refid" :"MUTH00000NATYF400063ST01HINOSHUOIN",
    "last_upt_ts": "2025-03-25 12:48:07.240",
    "duration_in_minutes" :20,
    "leg" : "Request send to BOU"
    },
    {
    "refid": "MUTH00000NATYF400063ST01HINOSHUOIN",
    "last_upt_ts": "2025-03-25 12:48:07.240",
    "duration_in_minutes" :20,
    "leg" : "Response received from BOU"
    },
    {
    "refid": "MUTH00000NATYF400063ST01HINOSHUOIN",
    "last_upt_ts": "2025-03-25 12:48:07.240",
    "duration_in_minutes" :20,
    "leg" :"Request Recieved from COU"
    
    }
]
const [showModal, setShowModal] = useState(false);

const handleTriggerClick = () => {
  setShowModal(true);
  // setTimeout(setShowModal(false),3000);
  setTimeout(() => {
    setShowModal(false);
  }, 4000);
};

const closeModal = () => {
  setShowModal(false);
};
return (
    <div className="stuckdashboardpage">
<div className="images-container">
<h1 style={{marginLeft:"29px",marginTop:"10px"}}>Abnormal Transactions</h1>
    </div>
    <table id="stucktable"> 
        <thead>
          <tr>
            <th>Reference ID</th>
            <th>Last Update Timestamp</th>
            <th>Duration (minutes)</th>
            <th>Leg</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {stuckData.map((item, index) => (
            <tr key={index}>
              <td>{item.refid}</td>
              <td>{item.last_upt_ts}</td>
              <td>{item.duration_in_minutes}</td>
              <td>{item.leg}</td>
              <td><a id="triggerBtn" onClick={handleTriggerClick}>Trigger</a></td>
            </tr>
          ))}
        </tbody>
      </table>
      {showModal && (
        <div className="modal-overlay1" onClick={closeModal}>
          <div className="modal-content1" onClick={(e) => e.stopPropagation()}>
            <h2>✅ Mail Sent!</h2>
            <p>The mail has been successfully sent to the support team.</p>
            <button className="close-btn" onClick={closeModal}>Close</button>
          </div>
        </div>
      )}

    </div>
);
};

export default StuckDashboard;
