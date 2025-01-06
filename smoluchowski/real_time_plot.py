import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import subprocess

def create_renderer(subprocess_cmd, accessor, on_click_command = False):
    class Renderer(tk.Tk):
        def __init__(self, subprocess_cmd, accessor):
            super().__init__()

            self.title("Smoluchowski Diffusion Real-Time Simulation")
            self.accessor  = accessor
            xlabel = "z"
            ylabel = "r"
            zlabel = "concentration"


            self.figure = plt.figure()
            self.shared_memory_array = self.accessor()

            Y,X = np.shape(self.shared_memory_array)
            aspect_ratio = X/Y
            if aspect_ratio>=1:
                y_scale = 1
                x_scale = 1/aspect_ratio
                top = 0.5
                right = top/aspect_ratio
            else:
                x_scale = 1
                y_scale = aspect_ratio
                right = 0.5
                top = right*aspect_ratio

            ax = self.figure.add_gridspec(
                top=1-top, right=1-right,
                #left = 0.1, bottom = 0.1,
                ).subplots()


            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            self.main_ax = ax

            self.ax_x = ax.inset_axes([0, 1.05, 1, y_scale])
            self.ax_y = ax.inset_axes([1.05, 0, x_scale, 1])
            self.cbar_ax = self.ax_y.inset_axes([0.1, 1.15, 0.85, 0.05])
            self.ax_x.set_xticks([])   
            self.ax_y.set_yticks([])

            self.main_ax.set(aspect=1)
            self.ax_x.set_ylabel(zlabel)
            self.ax_y.set_xlabel(zlabel)

            # Display the initial image
            self.im = self.main_ax.imshow(
                self.shared_memory_array, 
                cmap="gnuplot", 
                vmin=0, 
                vmax=1,
                origin='lower'  # Ensure the origin matches the array indexing
            )
            self.x_profile, = self.ax_x.plot(self.shared_memory_array[0])
            self.y_profile, = self.ax_y.plot(self.shared_memory_array[:,int(X/2)], np.arange(Y))

            #self.contour = ax.contour(self.shared_memory_array, levels = [0.05, 0.25, 0.5, 0.75, 0.95], colors= "white", linewidths = 0.3)

            self.colorbar = self.figure.colorbar(self.im, ax=self.main_ax, cax = self.cbar_ax, 
                            orientation = "horizontal",
                            )
            self.colorbar.set_label(zlabel, labelpad = -40)

            fig_size=7
            if aspect_ratio>=1:
                self.figure.set_size_inches(fig_size,fig_size/aspect_ratio+0.5)
            else:
                self.figure.set_size_inches(fig_size*aspect_ratio,fig_size+0.5)

            # text = ax_x.text(
            #     1.05, 0.95, 
            #     f"{xlabel}={x_arr[vline_x]}\n{ylabel}={y_arr[hline_y]}\n{zlabel}={zvalue:.3f}", 
            #     transform=ax_x.transAxes, 
            #     ha="left", 
            #     va = "top", 
            #     fontsize=13
            #     )

            # Embed the Matplotlib figure in Tkinter
            self.canvas = FigureCanvasTkAgg(self.figure, self)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Force an initial draw to ensure the plot is rendered correctly
            self.canvas.draw()

            # Connect the mouse click event
            self.canvas.mpl_connect("button_press_event", self.on_click)

            # Handle window close event
            self.protocol("WM_DELETE_WINDOW", self.on_closing)

            # Start the subprocess
            self.process = subprocess.Popen(subprocess_cmd)

            # Start updating the plot
            self.update_plot()

        if on_click_command:
            def on_click(self, event):
                """
                Event handler for mouse clicks on the plot.
                Sets a 10x10 rectangle centered at the click to 1.0 in the shared memory array.
                """
                if event.inaxes != self.main_ax:
                    print("Click outside the plot area.")
                    return  # Ignore clicks outside the plot

                # Get the data coordinates from the click
                ydata, xdata = event.xdata, event.ydata

                # Convert to integer indices
                i = int(np.round(ydata))
                j = int(np.round(xdata))

                # Define the size of the rectangle
                rect_size = 10
                half_size = rect_size // 2

                # Get array dimensions
                cols, rows = self.shared_memory_array.shape

                # Calculate the start and end indices, ensuring they are within bounds
                i_start = max(i - half_size, 0)
                i_end = min(i + half_size, rows)
                j_start = max(j - half_size, 0)
                j_end = min(j + half_size, cols)

                # Debug: Print the clicked position and the region to be updated
                #print(f"Clicked at data coordinates ({xdata:.2f}, {ydata:.2f}) -> array indices ({i}, {j}).")
                #print(f"Updating region: rows {i_start}-{i_end}, cols {j_start}-{j_end}")

                # Update the shared memory array
                self.shared_memory_array[j_start:j_end, i_start:i_end] += 10.0
        else:
            on_click = None

        def update_plot(self):
            # Check if the subprocess has finished
            if self.process.poll() is not None:
                self.destroy()  # Close the Tkinter application
                return

            # for coll in self.contour.get_paths(): 
            #     plt.gca().collections.remove(coll)
            #self.contour.remove()
            #self.contour = self.ax.contour(self.shared_memory_array, levels = [0.05, 0.25, 0.5, 0.75, 0.95], colors= "white", linewidths = 0.3)

            # Access shared memory data
            try:
                self.shared_memory_array = self.accessor()
            except Exception as e:
                print(f"Error accessing shared memory: {e}")
                self.destroy()
                return

            # Update imshow data
            self.im.set_data(self.shared_memory_array)
            # Optionally update color limits if dynamic
            self.im.set_clim(vmin=self.shared_memory_array.min(), vmax=self.shared_memory_array.max())

            Y,X = np.shape(self.shared_memory_array)
            self.x_profile.set_ydata(self.shared_memory_array[0])
            self.ax_x.set_ylim(self.shared_memory_array.min(), self.shared_memory_array.max())
            self.y_profile.set_xdata(self.shared_memory_array[:,int(X/2)])
            self.ax_y.set_xlim(self.shared_memory_array.min(), self.shared_memory_array.max())

            # Redraw the canvas
            self.canvas.draw_idle()  # Use draw_idle for better performance

            # Schedule the next update
            self.after(100, self.update_plot)  # Update every 100ms

        def on_closing(self):
            """
            Handles the window closing event.
            Terminates the subprocess and closes the shared memory allocator.
            """
            print("Closing application...")
            if self.process.poll() is None:
                print("Terminating subprocess...")
                self.process.terminate()
                self.process.wait()
            self.destroy()

    renderer = Renderer(subprocess_cmd, accessor)
    return renderer