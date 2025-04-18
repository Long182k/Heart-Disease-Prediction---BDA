import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  FaHeartbeat,
  FaUser,
  FaChartBar,
  FaSignOutAlt,
  FaSignInAlt,
  FaUserPlus,
  FaShieldAlt,
  FaCaretDown,
} from "react-icons/fa";

const Navbar = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState("");
  const [userRole, setUserRole] = useState("");
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem("cardio_auth_token");
    const user = JSON.parse(localStorage.getItem("cardio_user") || "{}");

    if (token && user.username) {
      setIsLoggedIn(true);
      setUsername(user.username);
      setUserRole(user.role || "user");
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setIsLoggedIn(false);
    setDropdownOpen(false);
    navigate("/login");
  };

  const toggleDropdown = () => {
    setDropdownOpen(!dropdownOpen);
  };

  return (
    <nav className="border-b bg-background">
      <div className="container flex justify-between items-center px-4 py-4 mx-auto">
        <Link to="/" className="flex items-center space-x-2">
          <FaHeartbeat className="text-2xl text-primary" />
          <span className="text-xl font-bold">CardioGuardian</span>
        </Link>

        <div className="flex items-center space-x-6">
          <Link to="/predict" className="text-sm hover:text-primary">
            Predict
          </Link>
          <Link to="/metrics" className="text-sm hover:text-primary">
            <div className="flex items-center">
              <FaChartBar className="mr-1" />
              Model Metrics
            </div>
          </Link>

          {isLoggedIn ? (
            <div className="relative">
              <button
                onClick={toggleDropdown}
                className="flex items-center px-3 py-2 text-sm bg-gray-100 rounded-md hover:text-primary"
              >
                <FaUser className="mr-2" />
                Hi {username}
                <FaCaretDown className="ml-2" />
              </button>

              {dropdownOpen && (
                <div className="absolute right-0 z-10 py-1 mt-2 w-48 bg-white rounded-md border shadow-lg">
                  {userRole === "admin" && (
                    <Link
                      to="/admin"
                      className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                      onClick={() => setDropdownOpen(false)}
                    >
                      <div className="flex items-center">
                        <FaShieldAlt className="mr-2" />
                        Admin Dashboard
                      </div>
                    </Link>
                  )}
                  <Link
                    to="/profile"
                    className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    onClick={() => setDropdownOpen(false)}
                  >
                    <div className="flex items-center">
                      <FaUser className="mr-2" />
                      My Profile
                    </div>
                  </Link>
                  <button
                    onClick={handleLogout}
                    className="block px-4 py-2 w-full text-sm text-left text-gray-700 hover:bg-gray-100"
                  >
                    <div className="flex items-center">
                      <FaSignOutAlt className="mr-2" />
                      Logout
                    </div>
                  </button>
                </div>
              )}
            </div>
          ) : (
            <>
              <Link to="/login" className="text-sm hover:text-primary">
                <div className="flex items-center">
                  <FaSignInAlt className="mr-1" />
                  Login
                </div>
              </Link>
              <Link to="/register" className="text-sm hover:text-primary">
                <div className="flex items-center">
                  <FaUserPlus className="mr-1" />
                  Register
                </div>
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
