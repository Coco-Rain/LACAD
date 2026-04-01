import cadquery as cq

result = (
cq.Workplane("XZ")
.circle(15)
.extrude(60)
.transformed(rotate=(90, 0, 0), offset=(0, 0, 25.5))
.split(keepTop=True)
.translate((0,30,0))
)
cq.exporters.export(result, 'GT.stl')