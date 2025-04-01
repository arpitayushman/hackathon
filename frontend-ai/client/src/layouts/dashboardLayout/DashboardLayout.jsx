import { Outlet, useNavigate } from "react-router-dom";
import "./dashboardLayout.css";

import { useEffect } from "react";
import ChatList from "../../components/chatList/ChatList";

const DashboardLayout = () => {
 







  return (
    <div className="dashboardLayout">
      <div className="content">
        <Outlet />
      </div>
    </div>
  );
};

export default DashboardLayout;
