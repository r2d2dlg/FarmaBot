USE [master]
GO
/****** Object:  Database [ChatbotFarmacia]    Script Date: 4/4/2025 9:41:19 AM ******/
CREATE DATABASE [ChatbotFarmacia]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'ChatbotFarmacia', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.MSSQLSERVER\MSSQL\DATA\ChatbotFarmacia.mdf' , SIZE = 73728KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'ChatbotFarmacia_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.MSSQLSERVER\MSSQL\DATA\ChatbotFarmacia_log.ldf' , SIZE = 8192KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT, LEDGER = OFF
GO
ALTER DATABASE [ChatbotFarmacia] SET COMPATIBILITY_LEVEL = 160
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [ChatbotFarmacia].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [ChatbotFarmacia] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET ARITHABORT OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [ChatbotFarmacia] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [ChatbotFarmacia] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET  ENABLE_BROKER 
GO
ALTER DATABASE [ChatbotFarmacia] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [ChatbotFarmacia] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET RECOVERY FULL 
GO
ALTER DATABASE [ChatbotFarmacia] SET  MULTI_USER 
GO
ALTER DATABASE [ChatbotFarmacia] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [ChatbotFarmacia] SET DB_CHAINING OFF 
GO
ALTER DATABASE [ChatbotFarmacia] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [ChatbotFarmacia] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [ChatbotFarmacia] SET DELAYED_DURABILITY = DISABLED 
GO
ALTER DATABASE [ChatbotFarmacia] SET ACCELERATED_DATABASE_RECOVERY = OFF  
GO
EXEC sys.sp_db_vardecimal_storage_format N'ChatbotFarmacia', N'ON'
GO
ALTER DATABASE [ChatbotFarmacia] SET QUERY_STORE = ON
GO
ALTER DATABASE [ChatbotFarmacia] SET QUERY_STORE (OPERATION_MODE = READ_WRITE, CLEANUP_POLICY = (STALE_QUERY_THRESHOLD_DAYS = 30), DATA_FLUSH_INTERVAL_SECONDS = 900, INTERVAL_LENGTH_MINUTES = 60, MAX_STORAGE_SIZE_MB = 1000, QUERY_CAPTURE_MODE = AUTO, SIZE_BASED_CLEANUP_MODE = AUTO, MAX_PLANS_PER_QUERY = 200, WAIT_STATS_CAPTURE_MODE = ON)
GO
USE [ChatbotFarmacia]
GO
/****** Object:  Table [dbo].[inventory_costa_del_este]    Script Date: 4/4/2025 9:41:19 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[inventory_costa_del_este](
	[Brand Name] [varchar](30) NOT NULL,
	[Generic Name] [varchar](30) NOT NULL,
	[Inventory] [int] NULL,
 CONSTRAINT [PK_inventory_costa_del_este] PRIMARY KEY CLUSTERED 
(
	[Brand Name] ASC,
	[Generic Name] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[inventory_el_dorado]    Script Date: 4/4/2025 9:41:19 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[inventory_el_dorado](
	[Brand Name] [varchar](30) NOT NULL,
	[Generic Name] [varchar](30) NOT NULL,
	[Inventory] [int] NULL,
 CONSTRAINT [PK_inventory_el_dorado] PRIMARY KEY CLUSTERED 
(
	[Brand Name] ASC,
	[Generic Name] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[inventory_san_francisco]    Script Date: 4/4/2025 9:41:19 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[inventory_san_francisco](
	[Brand Name] [varchar](30) NOT NULL,
	[Generic Name] [varchar](30) NOT NULL,
	[Inventory] [int] NULL,
 CONSTRAINT [PK_inventory_san_francisco] PRIMARY KEY CLUSTERED 
(
	[Brand Name] ASC,
	[Generic Name] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[inventory_chorrera]    Script Date: 4/4/2025 9:41:19 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[inventory_chorrera](
	[Brand Name] [varchar](30) NOT NULL,
	[Generic Name] [varchar](30) NOT NULL,
	[Inventory] [int] NULL,
 CONSTRAINT [PK_inventory_chorrera] PRIMARY KEY CLUSTERED 
(
	[Brand Name] ASC,
	[Generic Name] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[inventory_david]    Script Date: 4/4/2025 9:41:19 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[inventory_david](
	[Brand Name] [varchar](30) NOT NULL,
	[Generic Name] [varchar](30) NOT NULL,
	[Inventory] [int] NULL,
 CONSTRAINT [PK_inventory_david] PRIMARY KEY CLUSTERED 
(
	[Brand Name] ASC,
	[Generic Name] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[v_inventory]    Script Date: 4/4/2025 9:41:19 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [dbo].[v_inventory] AS
SELECT 'Costa del Este' AS Store, * FROM dbo.inventory_costa_del_este
UNION ALL
SELECT 'El Dorado', * FROM dbo.inventory_el_dorado
UNION ALL
SELECT 'San Francisco', * FROM dbo.inventory_san_francisco
UNION ALL
SELECT 'Chorrera', * FROM dbo.inventory_chorrera
UNION ALL
SELECT 'David', * FROM dbo.inventory_david;
GO
/****** Object:  Table [dbo].[inventory]    Script Date: 4/4/2025 9:41:19 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[inventory](
	[Brand Name] [varchar](30) NOT NULL,
	[Generic Name] [varchar](30) NOT NULL,
	[Inventory] [int] NULL,
 CONSTRAINT [PK_inventory] PRIMARY KEY CLUSTERED 
(
	[Brand Name] ASC,
	[Generic Name] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[InventoryChangeLog]    Script Date: 4/4/2025 9:41:19 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[InventoryChangeLog](
	[ChangeID] [int] IDENTITY(1,1) NOT NULL,
	[TableName] [nvarchar](128) NULL,
	[BrandName] [nvarchar](100) NULL,
	[GenericName] [nvarchar](100) NULL,
	[OldInventory] [int] NULL,
	[NewInventory] [int] NULL,
	[ChangeDate] [datetime] NULL,
PRIMARY KEY CLUSTERED 
(
	[ChangeID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Medicines]    Script Date: 4/4/2025 9:41:19 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Medicines](
	[Generic Name] [varchar](max) NULL,
	[Brand Name 1] [varchar](max) NULL,
	[Brand Name 2] [varchar](max) NULL,
	[Brand Name 3] [varchar](max) NULL,
	[Brand Name 4] [varchar](max) NULL,
	[Brand Name 5] [varchar](max) NULL,
	[Brand Name 6] [varchar](max) NULL,
	[Uses] [varchar](max) NULL,
	[Side Effects (Common)] [varchar](max) NULL,
	[Side Effects (Rare)] [varchar](max) NULL,
	[Similar Drugs] [varchar](max) NULL,
	[Prescription] [bit] NULL,
	[MedicineID] [int] IDENTITY(1,1) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[MedicineID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Stores]    Script Date: 4/4/2025 9:41:19 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Stores](
	[StoreID] [int] IDENTITY(1,1) NOT NULL,
	[StoreName] [nvarchar](100) NOT NULL,
	[InventoryTableName] [nvarchar](128) NULL,
	[Location] [nvarchar](255) NULL,
	[Address] [nvarchar](255) NULL,
PRIMARY KEY CLUSTERED 
(
	[StoreID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
UNIQUE NONCLUSTERED 
(
	[StoreName] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
ALTER TABLE [dbo].[InventoryChangeLog] ADD  DEFAULT (getdate()) FOR [ChangeDate]
GO
/****** Object:  StoredProcedure [dbo].[UpdateAggregateInventory]    Script Date: 4/4/2025 9:41:19 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE   PROCEDURE [dbo].[UpdateAggregateInventory]
AS
BEGIN
    -- Clear existing data in dbo.inventory
    TRUNCATE TABLE dbo.inventory;

    -- Insert aggregated data from all inventory tables
    INSERT INTO dbo.inventory ([Brand Name], [Generic Name], [Inventory])
    SELECT [Brand Name], [Generic Name], SUM([Inventory]) AS TotalInventory
    FROM (
        SELECT [Brand Name], [Generic Name], [Inventory] FROM dbo.inventory_chorrera
        UNION ALL
        SELECT [Brand Name], [Generic Name], [Inventory] FROM dbo.inventory_costa_del_este
        UNION ALL
        SELECT [Brand Name], [Generic Name], [Inventory] FROM dbo.inventory_david
        UNION ALL
        SELECT [Brand Name], [Generic Name], [Inventory] FROM dbo.inventory_el_dorado
        UNION ALL
        SELECT [Brand Name], [Generic Name], [Inventory] FROM dbo.inventory_san_francisco
    ) AS AllInventories
    GROUP BY [Brand Name], [Generic Name];
END;
GO
USE [master]
GO
ALTER DATABASE [ChatbotFarmacia] SET  READ_WRITE 
GO

-- Insert store data with addresses
INSERT INTO [dbo].[Stores] ([StoreName], [InventoryTableName], [Location], [Address]) VALUES
('Chorrera', 'inventory_chorrera', 'Panamá Oeste - La Chorrera', 'Centro Comercial Plaza Milenio, Calle 10, Barrio Colón, La Chorrera'),
('Costa del Este', 'inventory_costa_del_este', 'Panama City - Costa del Este', 'Town Center, Boulevard Costa del Este & Avenida Paseo del Mar, Costa del Este'),
('David', 'inventory_david', 'Chiriquí - David', 'Plaza Terronal, Avenida 3ra Este & Calle Central, David, Chiriquí'),
('El Dorado', 'inventory_el_dorado', 'Panama City - El Dorado', 'Centro Comercial El Dorado, Av. Ricardo J. Alfaro, Local 24, Bethania'),
('San Francisco', 'inventory_san_francisco', 'Panama City - San Francisco', 'Plaza Belen, calle 66 Este & C. 66 Este, Panamá');
GO
