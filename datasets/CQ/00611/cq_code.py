import cadquery as cq

path = cq.Workplane("XZ").rect(100, 50)
result = cq.Workplane("XY", origin=(-50, 0, 0)).circle(10).sweep(path)
cq.exporters.export(result, 'GT.stl')