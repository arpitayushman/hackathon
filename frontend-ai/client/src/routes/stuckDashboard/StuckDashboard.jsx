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
    "leg" :2
    
    },
    {
    "refid" :"MUTH00000NATYF400063ST01HINOSHUOIN",
    "last_upt_ts": "2025-03-25 12:48:07.240",
    "duration_in_minutes" :20,
    "leg" : 2
    },
    {
    "refid": "MUTH00000NATYF400063ST01HINOSHUOIN",
    "last_upt_ts": "2025-03-25 12:48:07.240",
    "duration_in_minutes" :20,
    "leg" : 3
    },
    {
    "refid": "MUTH00000NATYF400063ST01HINOSHUOIN",
    "last_upt_ts": "2025-03-25 12:48:07.240",
    "duration_in_minutes" :20,
    "leg" :1
    
    }
]


return (
    <div className="stuckdashboardpage">
<div className="images-container">
<h1>Stuck Transactions</h1>
    </div>
    <table>
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
              <td><a  id="triggerBtn" href="http://localhost:8084/">Trigger</a></td>
            </tr>
          ))}
        </tbody>
      </table>


    </div>
);
};

export default StuckDashboard;
