import pygame
import os
import math
import numpy as np

class Renderer:
    def __init__(self, width, height):
        pygame.init()
        pygame.font.init()

        self.width = width
        self.height = height
        self.window = None
        self.clock = None
        
        current_dir = os.path.dirname(__file__)
        self.assets_dir = os.path.join(current_dir, "..", "assets")
        
        self.images = {}
        self._load_assets()

    def _load_assets(self):
        self._load_image("jet_red", "plane_red.png")
        self._load_image("jet_blue", "plane_blue.png")
        self._load_image("helicopter", "helicopter.png")
        self._load_image("runway_red", "runway_red.png")
        self._load_image("runway_blue", "runway_blue.png")
        self._load_image("helipad", "helipad.png")
        self._load_image("background", "background.png")

    def _load_image(self, key, filename):
        path = os.path.join(self.assets_dir, filename)
        if os.path.exists(path):
            try:
                img = pygame.image.load(path)
                self.images[key] = img
            except Exception:
                self.images[key] = None
        else:
            self.images[key] = None

    def init_window(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

    def draw(self, mode, planes, zones, wind_vector=None):
        if self.window is None and mode == "human":
            self.init_window()
            
        canvas = pygame.Surface((self.width, self.height))
        
        if self.images["background"]:
            bg = pygame.transform.scale(self.images["background"], (self.width, self.height))
            canvas.blit(bg, (0, 0))
        else:
            canvas.fill((34, 139, 34)) 

        for zone in zones:
            self._draw_zone(canvas, zone)

        for plane in planes:
            self._draw_plane(canvas, plane)

        if wind_vector is not None:
            start_pos = (50, 50)
            end_pos = (
                50 + wind_vector[0] * 40, 
                50 + wind_vector[1] * 40
            )
            
            pygame.draw.line(canvas, (255, 255, 255), start_pos, end_pos, 3)
            pygame.draw.circle(canvas, (200, 200, 200), start_pos, 3)

        if mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(30)
        
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _draw_zone(self, surface, zone):
        img = self.images.get(zone.type)
        scale_factor = 1.5
       
        if img:
            new_width = int(img.get_width() * scale_factor)
            new_height = int(img.get_height() * scale_factor)
            scaled_img = pygame.transform.scale(img, (new_width, new_height))
            
            angle_deg = -math.degrees(zone.angle) if zone.angle else 0
            rotated_img = pygame.transform.rotate(scaled_img, angle_deg)
            
            rect = rotated_img.get_rect(center=(zone.x, zone.y))
            surface.blit(rotated_img, rect)
        else:
            if zone.type == "runway_red":
                color = (200, 50, 50)
            elif zone.type == "runway_blue":
                color = (50, 50, 200)
            elif zone.type == "helipad":
                color = (100, 100, 100)
            else:
                color = (50, 50, 50)

            if zone.type == "helipad":
                pygame.draw.circle(surface, color, (int(zone.x), int(zone.y)), 30)
                pygame.draw.circle(surface, (255, 255, 255), (int(zone.x), int(zone.y)), 25, 2)
                
                if not pygame.font.get_init():
                    pygame.font.init()
                
                font = pygame.font.SysFont(None, 24)
                text = font.render("H", True, (255, 255, 255))
                text_rect = text.get_rect(center=(int(zone.x), int(zone.y)))
                surface.blit(text, text_rect)

                indicator_color = (50, 200, 50)
                pygame.draw.circle(surface, indicator_color, (int(zone.x), int(zone.y)), 5)
            else:
                rect_w, rect_h = 120, 40
                
                surf = pygame.Surface((rect_w, rect_h), pygame.SRCALPHA)
                surf.fill(color)
                pygame.draw.rect(surf, (255, 255, 255), (0, 0, rect_w, rect_h), 2)
                pygame.draw.line(surf, (255, 255, 255), (10, rect_h/2), (rect_w-10, rect_h/2), 2)
                
                angle_deg = -math.degrees(zone.angle)
                rotated_surf = pygame.transform.rotate(surf, angle_deg)
                new_rect = rotated_surf.get_rect(center=(zone.x, zone.y))
                surface.blit(rotated_surf, new_rect)

        if "runway" in zone.type:
            if "red" in zone.type:
                indicator_color = (255, 0, 0)
            elif "blue" in zone.type:
                indicator_color = (0, 0, 255)
            else:
                indicator_color = (255, 255, 255)

            pygame.draw.circle(surface, indicator_color, (int(zone.x), int(zone.y)), 5)

            arrow_length = 60
            start_x = zone.x - arrow_length * math.cos(zone.angle)
            start_y = zone.y - arrow_length * math.sin(zone.angle)
            
            pygame.draw.line(surface, indicator_color, (start_x, start_y), (zone.x, zone.y), 3)

            arrow_head_size = 15
            wing1_x = zone.x - arrow_head_size * math.cos(zone.angle + 0.5)
            wing1_y = zone.y - arrow_head_size * math.sin(zone.angle + 0.5)
            wing2_x = zone.x - arrow_head_size * math.cos(zone.angle - 0.5)
            wing2_y = zone.y - arrow_head_size * math.sin(zone.angle - 0.5)

            pygame.draw.polygon(surface, indicator_color, [(zone.x, zone.y), (wing1_x, wing1_y), (wing2_x, wing2_y)])
            

    def _draw_plane(self, surface, plane):
        key = plane.type
        img = self.images.get(key)
        
        if img:
            angle_deg = -math.degrees(plane.heading)
            rotated_img = pygame.transform.rotate(img, angle_deg)
            rect = rotated_img.get_rect(center=(plane.x, plane.y))
            surface.blit(rotated_img, rect)
        else:
            if plane.type == "jet_red":
                color = (200, 50, 50)
            elif plane.type == "jet_blue":
                color = (50, 50, 200)
            else:
                color = (50, 200, 50)
            
            surf = pygame.Surface((30, 30), pygame.SRCALPHA)
            pygame.draw.polygon(surf, color, [(30, 15), (0, 0), (0, 30)])
            angle_deg = -math.degrees(plane.heading)
            rotated_surf = pygame.transform.rotate(surf, angle_deg)
            rect = rotated_surf.get_rect(center=(plane.x, plane.y))
            surface.blit(rotated_surf, rect)
            
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None