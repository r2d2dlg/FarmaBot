USE [ChatbotFarmacia]
GO

-- Add Address column if it doesn't exist
IF NOT EXISTS (
    SELECT * FROM sys.columns 
    WHERE object_id = OBJECT_ID(N'[dbo].[Stores]') 
    AND name = 'Address'
)
BEGIN
    ALTER TABLE [dbo].[Stores]
    ADD [Address] [nvarchar](500) NULL
END
GO

-- First, clear existing data
DELETE FROM [dbo].[Stores];
GO

-- Insert updated store data with new addresses (no accents)
INSERT INTO [dbo].[Stores] ([StoreName], [InventoryTableName], [Location], [Address]) VALUES
('Chorrera', 'inventory_chorrera', 'Panama Oeste - La Chorrera', 'Centro Comercial Plaza Milenio, Calle 10, Barrio Colon, La Chorrera'),
('Costa del Este', 'inventory_costa_del_este', 'Panama City - Costa del Este', 'Town Center, Boulevard Costa del Este y Avenida Paseo del Mar, Costa del Este'),
('David', 'inventory_david', 'Chiriqui - David', 'Plaza Terronal, Avenida 3ra Este y Calle Central, David, Chiriqui'),
('El Dorado', 'inventory_el_dorado', 'Panama City - El Dorado', 'Centro Comercial El Dorado, Av. Ricardo J. Alfaro, Local 24, Bethania'),
('San Francisco', 'inventory_san_francisco', 'Panama City - San Francisco', 'Plaza Belen, Calle 66 Este y Calle 66 Este, Panama');
GO

-- Update store addresses
UPDATE [dbo].[Stores]
SET [Address] = 'Centro Comercial Plaza Milenio, Calle 10, Barrio Colon, La Chorrera'
WHERE [StoreName] = 'Chorrera';

UPDATE [dbo].[Stores]
SET [Address] = 'Town Center, Boulevard Costa del Este y Avenida Paseo del Mar, Costa del Este'
WHERE [StoreName] = 'Costa del Este';

UPDATE [dbo].[Stores]
SET [Address] = 'Plaza Terronal, Avenida 3ra Este y Calle Central, David, Chiriqui'
WHERE [StoreName] = 'David';

UPDATE [dbo].[Stores]
SET [Address] = 'Centro Comercial El Dorado, Av. Ricardo J. Alfaro, Local 24, Bethania'
WHERE [StoreName] = 'El Dorado';

UPDATE [dbo].[Stores]
SET [Address] = 'Plaza Belen, Calle 66 Este y Calle 66 Este, Panama'
WHERE [StoreName] = 'San Francisco';
GO 