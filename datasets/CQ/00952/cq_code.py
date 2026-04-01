import cadquery as cq

result = (
cq.Workplane("XY")
.sphere(10, angle1=45, angle2=90, angle3=90)
)
cq.exporters.export(result, 'GT.stl')